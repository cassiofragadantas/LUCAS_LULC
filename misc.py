import math
import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
import pandas as pd
import os
from collections import OrderedDict
from matplotlib import pyplot as plt
import itertools

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Oranges,
                          filename = None, normalized = False, text = True):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(cm.shape[1])
    plt.xticks(tick_marks, rotation=45)
    ax = plt.gca()
    ax.set_xticklabels((ax.get_xticks() +1).astype(str))
    plt.yticks(tick_marks)

    thresh = cm.max() / 2.
    form = '.2f' if normalized else 'd'
    if text:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], form),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if filename is not None:
        plt.savefig(filename + '.png', dpi=300, bbox_inches='tight')

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

def loadData(data_path, suffix='prime', pred_level=2, loo_region=None):
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
    climate_test = LU22_test['climate']

    # Lat-Long encoding (inspired by Baudoux2021 and Bellet2024 PhD Thesis)
    # see https://github.com/LBaudoux/Unet_LandCoverTranslator
    geo_enc_train = positional_encoding(LU22_train)
    geo_enc_test = positional_encoding(LU22_test)

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

    # Re-shuffle train-test split by selecting a held-out region for test
    if loo_region is not None:
        dataset_all = pd.concat([LU22_train, LU22_test], ignore_index=True)
        y_all = pd.concat([y_train, y_test], ignore_index=True)
        geo_enc_all = np.concatenate([geo_enc_train, geo_enc_test])
        climate_all = np.concatenate([climate_train, climate_test])

        # idx_climate = (climate_all == loo_region)
        idx_climate = np.isin(climate_all, loo_region) # Get indexes for held-out climate zone (could be more than one)
        
        LU22_train, LU22_test = dataset_all[~idx_climate], dataset_all[idx_climate]
        y_train, y_test = y_all[~idx_climate], y_all[idx_climate]
        geo_enc_train, geo_enc_test = geo_enc_all[~idx_climate], geo_enc_all[idx_climate]
        climate_train, climate_test = climate_all[~idx_climate], climate_all[idx_climate]

    # Print train and test set distribution per climate region
    unique_climates, counts_climate = np.unique(np.array(climate_train, dtype=str), return_counts=True)
    print(f'Climate regions train: {unique_climates}')
    print(f'Samples per climate region train: {counts_climate}')
    unique_climates, counts_climate = np.unique(np.array(climate_test, dtype=str), return_counts=True)
    print(f'Climate regions test: {unique_climates}')
    print(f'Samples per climate region test: {counts_climate}')

    return LU22_train.to_numpy(), y_train.to_numpy(), LU22_test.to_numpy(), y_test.to_numpy(), \
            climate_train, geo_enc_train, geo_enc_test

def positional_encoding(dataset):
    d = 128
    d_i=np.arange(0,int(d/4))
    freq=1/(10000**(2*d_i/d))
    x, y = dataset['long'], dataset['lat']
    # x,y=x/10000,y/10000
    geo_enc=np.zeros((x.shape[0],d))
    d2 = int(d/2)
    geo_enc[:,0:d2:2]  = np.sin(np.outer(x, freq))
    geo_enc[:,1:d2:2]  = np.cos(np.outer(x, freq))
    geo_enc[:,d2::2]   = np.sin(np.outer(y, freq))
    geo_enc[:,d2+1::2] = np.cos(np.outer(y, freq))
    print(geo_enc.shape)

    return geo_enc

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
        for data in dataloader:
            x_batch = data[0].to(device)
            y_batch = data[1].to(device)
            if len(data) == 3: # coordinates are provided
                coord = data[2].to(device)
                pred = model(x_batch, coord)[0]
            else:
                pred = model(x_batch)[0]
            pred_npy = np.argmax(pred.cpu().detach().numpy(), axis=1)
            tot_pred.append( pred_npy )
            tot_labels.append( y_batch.cpu().detach().numpy())
        tot_pred = np.concatenate(tot_pred)
        tot_labels = np.concatenate(tot_labels)
    
    return tot_pred, tot_labels


# Supervised contrastive classes definition

# 3C classes (C per domain + C for domain invariant)
def sim_dist_specifc_loss_spc(spec_emb, ohe_label, ohe_dom, scl, epoch):
    norm_spec_emb = nn.functional.normalize(spec_emb)
    hash_label = {}
    new_combined_label = []
    for v1, v2 in zip(ohe_label, ohe_dom):
        key = "%d_%d"%(v1,v2)
        if key not in hash_label:
            hash_label[key] = len(hash_label)
        new_combined_label.append( hash_label[key] )
    new_combined_label = torch.tensor(np.array(new_combined_label), dtype=torch.int64)
    return scl(norm_spec_emb, new_combined_label, epoch=epoch)

# C + D classes: C for domain invariant + 1 domain on domain spec embs
def sup_contra_Cplus2_classes(emb, ohe_label, ohe_dom, scl, epoch):
    norm_emb = nn.functional.normalize(emb)
    C = ohe_label.max() + 1
    D = ohe_dom.max()
    new_combined_label = [v1 if v2==D else C+v2 for v1, v2 in zip(ohe_label, ohe_dom)]
    new_combined_label = torch.tensor(np.array(new_combined_label), dtype=torch.int64)
    return scl(norm_emb, new_combined_label, epoch=epoch)


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
    def __init__(self, num_classes=8, num_domains=2):
        super(MLPDisentangleV4, self).__init__()

        self.inv = MLP(out_dim=num_classes)
        self.spec = MLP(out_dim=num_domains)

    def forward(self, x):
        classif, inv_emb, inv_emb_n1, inv_fc_feat = self.inv(x)
        classif_spec, spec_emb, spec_emb_n1, spec_fc_feat = self.spec(x)
        return classif, inv_emb, spec_emb, classif_spec, inv_emb_n1, spec_emb_n1, inv_fc_feat, spec_fc_feat


class MLPDisentanglePos(torch.nn.Module):
    def __init__(self, num_classes=8, pos_enc_dim=128, act_out=True, num_domains=2):
        super(MLPDisentanglePos, self).__init__()

        self.pos_enc = MLP(out_dim=pos_enc_dim, num_hidden_layers=1, act_out=act_out)
        self.inv = MLP(out_dim=num_classes)
        self.spec = MLP(out_dim=num_domains)

    def forward(self, x, coord):
        pos_enc = self.pos_enc(coord)[0]
        x = torch.concat((x,pos_enc),dim=1)
        classif, inv_emb, inv_emb_n1, inv_fc_feat = self.inv(x)
        classif_spec, spec_emb, spec_emb_n1, spec_fc_feat = self.spec(x)
        return classif, inv_emb, spec_emb, classif_spec, inv_emb_n1, spec_emb_n1, inv_fc_feat, spec_fc_feat


class MLPDisentanglePosDANN(torch.nn.Module):
    def __init__(self, num_classes=8, pos_enc_dim=128, act_out=True, num_domains=2, discr_nb_hidden=0):
        super(MLPDisentanglePosDANN, self).__init__()

        self.pos_enc = MLP(out_dim=pos_enc_dim, num_hidden_layers=1, act_out=act_out)
        self.inv = MLPenc()
        self.clf_task = FC_Classifier(num_classes=num_classes)
        self.spec = MLPenc()
        self.clf_dom = FC_Classifier(num_classes=num_domains)
        self.discr = FC_Classifier(num_classes=num_domains,num_hidden_layers=discr_nb_hidden)

    def forward(self, x, coord):
        pos_enc = self.pos_enc(coord)[0]
        x = torch.concat((x,pos_enc),dim=1)
        inv_emb, inv_emb_n1, inv_fc_feat = self.inv(x)
        classif = self.clf_task(inv_fc_feat)
        spec_emb, spec_emb_n1, spec_fc_feat = self.spec(x)
        classif_spec = self.clf_dom(spec_fc_feat)
        classif_discr = self.discr(grad_reverse(inv_fc_feat))
        return classif, inv_emb, spec_emb, classif_spec, inv_emb_n1, spec_emb_n1, inv_fc_feat, spec_fc_feat, classif_discr


class MLP(nn.Module):
    def __init__(self, out_dim, dropout_rate=0.5, num_hidden_layers=3, hidden_dim=256, act_out=False):
        super(MLP, self).__init__()

        self.flatten = nn.Flatten()

        layers = []
        layers.append(nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ))

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ))

        self.hidden_layers = nn.Sequential(*layers)

        self.out = nn.Linear(hidden_dim, out_dim)
        self.act_out = act_out

    def forward(self, inputs):
        emb = self.flatten(inputs)
        intermediate_outputs = []
        for layer in self.hidden_layers:
            emb = layer(emb)
            intermediate_outputs.append(emb)
        out = self.out(emb)
        if self.act_out:
            out = torch.sigmoid(out)
        return (out, *intermediate_outputs)


###########################################
# Code associated to DANN implementation
###########################################

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha=1.):
        ctx.alpha = alpha
        return x.view_as(x)
        #print(alpha)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output * -ctx.alpha
        return output, None

def grad_reverse(x,alpha=1.):
    return GradReverse.apply(x,alpha)


class MLPDisentangleDANN(torch.nn.Module):
    def __init__(self, num_classes=8, discr_nb_hidden=0):
        super(MLPDisentangleDANN, self).__init__()

        self.inv = MLPenc()
        self.clf_task = FC_Classifier(num_classes=num_classes)
        self.spec = MLPenc()
        self.clf_dom = FC_Classifier(num_classes=2)
        self.discr = FC_Classifier(num_classes=2,num_hidden_layers=discr_nb_hidden)

    def forward(self, x):
        inv_emb, inv_emb_n1, inv_fc_feat = self.inv(x)
        classif = self.clf_task(inv_fc_feat)
        spec_emb, spec_emb_n1, spec_fc_feat = self.spec(x)
        classif_spec = self.clf_dom(spec_fc_feat)
        classif_discr = self.discr(grad_reverse(inv_fc_feat))
        return classif, inv_emb, spec_emb, classif_spec, inv_emb_n1, spec_emb_n1, inv_fc_feat, spec_fc_feat, classif_discr

class MLPDANN(torch.nn.Module):
    def __init__(self, num_classes=8, discr_nb_hidden=2):
        super(MLPDANN, self).__init__()

        self.enc = MLPenc()
        self.clf = FC_Classifier(num_classes=num_classes)
        self.discr = FC_Classifier(num_classes=2, num_hidden_layers=discr_nb_hidden)

    def forward(self, x):
        _, _, out_emb = self.enc(x)
        classif = self.clf(out_emb)
        classif_discr = self.discr(grad_reverse(out_emb))
        return classif, classif_discr


class MLPenc(nn.Module):
    def __init__(self, hidden_dim=256, dropout_rate=0.5):
        super(MLPenc, self).__init__()
            
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


    def forward(self, inputs):
        inputs = self.flatten(inputs)
        emb = self.layer1(inputs)
        hidden_emb = self.layer2(emb) # OR: hidden_emb = emb
        out_emb = self.layer3(hidden_emb)
        return emb, hidden_emb, out_emb # No more intermediate features


class FC_Classifier(torch.nn.Module):
    def __init__(self, num_classes, num_hidden_layers=0, hidden_dim=256, drop_prob=0.5):
        super(FC_Classifier, self).__init__()

        # Hidden layers (if any)
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.LazyLinear(hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(p=drop_prob)
                )                
            )
        

        self.clf  = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, emb):

        for layer in self.hidden_layers:
            emb = layer(emb)

        return self.clf(emb)
