import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from collections import OrderedDict

def normalizeFeatures(data, feat_min=None, feat_max=None):
    # Normalize per feature min and max
    if feat_min is None:
        feat_min = data.min(axis=0)
    if feat_max is None:
        feat_max = data.max(axis=0)
    return (data - feat_min)/(feat_max - feat_min), feat_min, feat_max

def normalizeFeaturesdf(df, df_min=None, df_max=None):
    if df_min is None:
        df_min = df.min()
    if df_min is None:
        df_max = df.max()
    # Normalize per feature
    return (df - df_min)/(df_max - df_min), df_min, df_max

    # To avoid errors for non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())

def loadData(data_path, suffix='prime', pred_level=2):
    LU22_train=pd.read_csv(os.path.join(data_path,'LU22_final_train_' + suffix + '.csv'), sep=',')
    LU22_test= pd.read_csv(os.path.join(data_path, 'LU22_final_test_' + suffix + '.csv'), sep=',')

    # Remove NaN
    LU22_train.dropna()
    LU22_test.dropna()

    # For level 2 prediction: Select subset of crop samples
    if pred_level == 2:
        LU22_train=LU22_train.loc[LU22_train['Label_lev1_code'] == 200]
        LU22_test=LU22_test.loc[LU22_test['Label_lev1_code'] == 200]

    # Climate region
    climate_train = LU22_train['climate']


    # Labels
    if pred_level == 1: # Lev1
        y_train= LU22_train['Label_lev1_code'].copy()
        y_test= LU22_test['Label_lev1_code'].copy()
    else: # Lev2 
        y_train= LU22_train['Label_lev2_code'].copy()
        y_test= LU22_test['Label_lev2_code'].copy() 
    # Map labels from 0 to n_classes-1
    label_mapping = dict()
    for k, label in enumerate(y_train.unique()):
        y_train[y_train==label] = k
        y_test[y_test==label] = k
        label_mapping[k] = label
    print(f'Label mapping: {label_mapping}')
    # Class counts
    _, counts_train = np.unique(y_train, return_counts=True)
    _, counts_test = np.unique(y_test, return_counts=True)
    print(f'Samples per class train: {counts_train}')
    print(f'Samples per class test : {counts_test}')

    # Select subset of important features
    features= pd.read_csv(os.path.join(data_path,'LU22_imp_features_' + suffix + '.csv'), sep=',')
    LU22_train= LU22_train[features.iloc[:,0]]
    LU22_test= LU22_test[features.iloc[:,0]]

    return LU22_train.to_numpy(), y_train.to_numpy(), LU22_test.to_numpy(), y_test.to_numpy(), climate_train

def cumulate_EMA(model, ema_weights, alpha):
    current_weights = OrderedDict()
    current_weights_npy = OrderedDict()
    state_dict = model.state_dict()
    for k in state_dict:
        current_weights_npy[k] = state_dict[k].cpu().detach().numpy()

    if ema_weights is not None:
        for k in state_dict:
            current_weights_npy[k] = alpha * ema_weights[k].cpu().detach().numpy() + (1-alpha) * current_weights_npy[k]

    for k in state_dict:
        current_weights[k] = torch.tensor( current_weights_npy[k] )

    return current_weights

def evaluation(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        tot_pred = []
        tot_labels = []
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)[0]
            pred_npy = np.argmax(pred.cpu().detach().numpy(), axis=1)
            tot_pred.append( pred_npy )
            tot_labels.append( y_batch.cpu().detach().numpy())
        tot_pred = np.concatenate(tot_pred)
        tot_labels = np.concatenate(tot_labels)
    
    return tot_pred, tot_labels

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, min_tau=.07, max_tau=1., t_period=50, eps=1e-7):
    #def __init__(self, temperature=1., min_tau=.07, max_tau=1., t_period=50, eps=1e-7):
        """
        Implementation of the loss described in the paper Supervised Contrastive Learning :
        https://arxiv.org/abs/2004.11362

        :param temperature: int
        """
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.min_tau = min_tau
        self.max_tau = max_tau
        self.t_period = t_period
        self.eps = eps

    def forward(self, projections, targets, epoch=1):
        """
        :param projections: torch.Tensor, shape [batch_size, projection_dim]
        :param targets: torch.Tensor, shape [batch_size]
        :return: torch.Tensor, scalar
        """
        device = torch.device("cuda") if projections.is_cuda else torch.device("cpu")
        #temperature = self.min_tau + 0.5 * (self.max_tau - self.min_tau) * (1 + torch.cos(torch.tensor(torch.pi * epoch / self.t_period )))
        

        dot_product = torch.mm(projections, projections.T)
        ### For stability issues related to matrix multiplications
        #dot_product = torch.clamp(dot_product, -1+self.eps, 1-self.eps)
        ####GEODESIC SIMILARITY
        #print(projections)
        #print( dot_product )
        #print( torch.acos(dot_product) / torch.pi )
        #dot_product = 1. - ( torch.acos(dot_product) / torch.pi )

        dot_product_tempered = dot_product / self.temperature
        
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        stab_max, _ = torch.max(dot_product_tempered, dim=1, keepdim=True)
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - stab_max.detach() ) + 1e-5
        )

        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)

        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))

        #### FILTER OUT POSSIBLE NaN PROBLEMS #### 
        mdf = cardinality_per_samples!=0
        cardinality_per_samples = cardinality_per_samples[mdf]
        log_prob = log_prob[mdf]
        mask_combined = mask_combined[mdf]
        #### #### #### #### #### #### #### #### #### 

        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)
        return supervised_contrastive_loss


class MLPDisentangleV4(torch.nn.Module):
    def __init__(self, num_classes=8):
        super(MLPDisentangleV4, self).__init__()

        self.inv = MLP(num_classes=num_classes)
        self.spec = MLP(num_classes=2)        

    def forward(self, x):
        classif, inv_emb, inv_emb_n1, inv_fc_feat = self.inv(x)
        classif_spec, spec_emb, spec_emb_n1, spec_fc_feat = self.spec(x)
        return classif, inv_emb, spec_emb, classif_spec, inv_emb_n1, spec_emb_n1, inv_fc_feat, spec_fc_feat


class MLP(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5, hidden_dim=256):
        super(MLP, self).__init__()
            
        self.flatten = nn.Flatten()
        self.layer1 = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.clf = nn.Linear(hidden_dim, num_classes)           


    def forward(self, inputs):
        inputs = self.flatten(inputs)
        emb = self.layer1(inputs)
        hidden_emb = self.layer2(emb) # OR: hidden_emb = emb
        out_emb = self.layer3(hidden_emb)
        return self.clf(out_emb), emb, hidden_emb, out_emb # No more intermediate features
