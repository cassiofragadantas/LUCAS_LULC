import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
from tqdm import tqdm
import sys
import numpy as np
import pandas as pd
from os.path import join
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, classification_report 
from misc import MLPDisentangleV4, SupervisedContrastiveLoss, normalizeFeatures, loadData, cumulate_EMA, evaluation
from misc import sim_dist_specifc_loss_spc, sup_contra_Cplus2_classes
import torch.nn.functional as F


data_path = '../LU22_final_shared/'
epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 300
rng_seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'


######## Training loop
def trainMLPDis(train_data, train_label, climate_train, test_data, epochs,
             model_name = 'temp.pth', device = 'cpu'):

    training_batch_size = 128
    normalize_features = True

    train_data = train_data.copy().to_numpy()
    train_label = train_label.copy().to_numpy()

    # Normalize data
    if normalize_features:
        train_data, feat_min, feat_max = normalizeFeatures(train_data)
        test_data, _, _ = normalizeFeatures(test_data, feat_min, feat_max)

    # Map labels from 0 to n_classes-1
    label_mapping = dict()
    for k, label in enumerate(np.unique(train_label)):
        train_label[train_label==label] = k
        label_mapping[k] = label
    # print(f'Label mapping: {label_mapping}')

    # Separate by climate region
    idx_med = (climate_train == 'Mediterranean')
    train_target_data = train_data[idx_med]
    train_target_label = train_label[idx_med]

    train_source_data = train_data[~idx_med]
    train_source_label = train_label[~idx_med]

    train_data = np.concatenate([train_target_data, train_source_data],axis=0)
    train_label = np.concatenate([train_target_label, train_source_label],axis=0)
    train_domain_label = np.concatenate([np.zeros(train_target_label.shape[0]), np.ones(train_source_label.shape[0])], axis=0)

    x_train = torch.tensor(train_data, dtype=torch.float32)
    y_train = torch.tensor(train_label, dtype=torch.int64)
    dom_train = torch.tensor(train_domain_label, dtype=torch.int64)

    train_dataset = TensorDataset(x_train, y_train, dom_train)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=training_batch_size)

    print(f'Training model on {device}...')

    n_classes = len(np.unique(train_label))
    model = MLPDisentangleV4(n_classes).to(device)

    learning_rate = 0.0001
    loss_fn = nn.CrossEntropyLoss()
    scl = SupervisedContrastiveLoss()

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(epochs)):
        start = time.time()
        model.train()
        tot_loss, den = 0., 0.
        domain_loss, contra_tot_loss = 0., 0.
        
        for x_batch, y_batch, dom_batch in train_dataloader:
            if x_batch.shape[0] != training_batch_size:
                continue

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            dom_batch = dom_batch.to(device)

            optimizer.zero_grad()

            pred, inv_emb, spec_emb_d, spec_d_pred, inv_emb_n1, spec_emb_n1, inv_fc_feat, spec_fc_feat = model(x_batch)

            ohe_label = F.one_hot(y_batch,num_classes=n_classes).cpu().detach().numpy()
            ohe_dom = F.one_hot(dom_batch,num_classes=2).cpu().detach().numpy()

            ##### DOMAIN CLASSIFICATION #####
            loss_ce_spec_dom = loss_fn(spec_d_pred, dom_batch)

            ##### MIXED MAINFOLD & CONTRASTIVE LEARNING ####
            
            cl_labels_npy = y_batch.cpu().detach().numpy()
            #dummy_labels_npy = np.ones_like(cl_labels_npy) * n_classes
            #y_mix_labels = np.concatenate([ cl_labels_npy , dummy_labels_npy , cl_labels_npy],axis=0)
            y_mix_labels = np.concatenate([ cl_labels_npy , cl_labels_npy],axis=0)
            #y_mix_labels = np.concatenate([ cl_labels_npy , dummy_labels_npy],axis=0)
            
            
            #DOMAIN LABEL FOR DOMAIN-CLASS SPECIFIC EMBEDDING and DOMAIN SPECIFIC EMBEDDING IS 0 OR 1 
            spec_dc_dom_labels = dom_batch.cpu().detach().numpy()
            #DOMAIN LABEL FOR INV EMBEDDING IS 2
            inv_dom_labels = np.ones_like(spec_dc_dom_labels) * 2

            dom_mix_labels = np.concatenate([inv_dom_labels, spec_dc_dom_labels],axis=0)
            
            joint_embedding = torch.concat([inv_emb, spec_emb_d])
            mixdl_loss_supContraLoss = sim_dist_specifc_loss_spc(joint_embedding, y_mix_labels, dom_mix_labels, scl, epoch)
            
            joint_embedding_n1 = torch.concat([inv_emb_n1, spec_emb_n1])
            mixdl_loss_supContraLoss_n1 = sim_dist_specifc_loss_spc(joint_embedding_n1, y_mix_labels, dom_mix_labels, scl, epoch)

            joint_embedding_fc_feat = torch.concat([inv_fc_feat, spec_fc_feat])
            mixdl_loss_supContraLoss_fc = sim_dist_specifc_loss_spc(joint_embedding_fc_feat, y_mix_labels, dom_mix_labels, scl, epoch)
            # mixdl_loss_supContraLoss_fc = sup_contra_Cplus2_classes(joint_embedding_fc_feat, y_mix_labels, dom_mix_labels, scl, epoch)
            
            contra_loss = mixdl_loss_supContraLoss_fc #+ mixdl_loss_supContraLoss_n1 + mixdl_loss_supContraLoss 

            ####################################

            loss = loss_fn(pred, y_batch) + contra_loss + loss_ce_spec_dom


            loss.backward() # backward pass: backpropagate the prediction loss
            optimizer.step() # gradient descent: adjust the parameters by the gradients collected in the backward pass
            tot_loss+= loss.cpu().detach().numpy()
            contra_tot_loss+= contra_loss.cpu().detach().numpy()
            den+=1.

        end = time.time()
        # print("Epoch %d (%.2fs): train loss %.4f"%(epoch, (end-start), tot_loss/den))                
    
    torch.save(model.state_dict(), model_name)

    # Inference on all test data
    x_test = torch.tensor(test_data.copy().to_numpy(), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        y_pred_p1 = model(x_test)[0]
        y_pred_p1 = np.argmax(y_pred_p1.cpu().detach().numpy(), axis=1)
        y_pred_p1 = np.array([label_mapping[label] for label in y_pred_p1]) # Unmap predicted labels back to original convention

    return y_pred_p1


######## Load train data
# load prime data containing cloud free samples and all features
LU22_train_prime=pd.read_csv(join(data_path,'LU22_final_train_prime.csv'), sep=',')
# load gapfill data containing all samples and cloud free features
LU22_train_gapfill= pd.read_csv(join(data_path,'LU22_final_train_gapfill.csv'), sep=',')

LU22_train_prime.dropna()
LU22_train_gapfill.dropna()

null_counts_train_prime = LU22_train_prime.isnull().sum()
null_counts_train_gapfill = LU22_train_gapfill.isnull().sum()

######## Load test
LU22_test_prime= pd.read_csv(join(data_path,'LU22_final_test_prime.csv'), sep=',')
LU22_test_gapfill= pd.read_csv(join(data_path,'LU22_final_test_gapfill.csv'), sep=',')

LU22_test_prime.dropna()
LU22_test_gapfill.dropna()

null_counts_test_prime = LU22_test_prime.isnull().sum()
null_counts_test_gapfill = LU22_test_gapfill.isnull().sum()

######## load Import features 
# features for prime classification
f_prime= pd.read_csv(join(data_path,'LU22_imp_features_prime.csv'), sep=',')
# features for gapfill classification
f_gapfill= pd.read_csv(join(data_path,'LU22_imp_features_gapfill.csv'), sep=',')

####### Apply the classification

# define confusion matrix function

def CF(y_test, y_pred):
    cf=pd.DataFrame(confusion_matrix(y_test, y_pred))
    cr=classification_report(y_test, y_pred,output_dict=True)
    cr = pd.DataFrame(cr).transpose()
    cf.columns = cr.index[:-3]
    cf.index= cr.index[:-3]
    cf = pd.concat([cf, cr.iloc[:-3,:]], axis=1)
    cf1=cf.iloc[:,:-4]
    cf1=cf1.transpose()
    cf1['total']=cf1.sum(axis=1)
    cf1.loc['total']=cf1.sum(axis=0)
    cf1['precision (UA)']=cf['precision']
    cf1['f1-score']=cf['f1-score']
    cf1.loc['recall (PA)']=cf['recall'].transpose()
    return cf1



#### Prime Classification
yp_p=LU22_test_prime[['pointid','Label_lev1_code','Label_lev2_code']]

# Lev1
climate_train = LU22_train_prime['climate'].copy()
X_train_p1= LU22_train_prime[f_prime.iloc[:,0]] #select prime important features
y_train_p1= LU22_train_prime['Label_lev1_code']

X_test_p= LU22_test_prime[f_prime.iloc[:,0]] #select prime important features
y_test_p= LU22_test_prime['Label_lev1_code']

model_name = "model_MLPDis_p1_" + str(epochs) + 'ep_seed' + str(rng_seed) + '.pth'
y_pred_p1 = trainMLPDis(X_train_p1, y_train_p1, climate_train, X_test_p, epochs, model_name, device)

yp_1=pd.DataFrame(y_pred_p1, columns=['Label_lev1_pred'])
yp_1.index = yp_p.index
yp_p = pd.concat([yp_p, yp_1], axis=1, ignore_index=False)   

# eval lev 1
f1_Lev1_p1 = f1_score(y_test_p, y_pred_p1, average='weighted')
print(f'PRIME DATA')
print(f'F1 score level 1: {f1_Lev1_p1}')


# lev2

LU22_train_prime_2=LU22_train_prime.copy()
LU22_train_prime_2=LU22_train_prime_2.loc[LU22_train_prime_2['Label_lev1_code'] == 200]

climate_train = LU22_train_prime_2['climate'].copy()
X_train_p2=LU22_train_prime_2[f_prime.iloc[:,0]]
y_train_p2 = LU22_train_prime_2['Label_lev2_code']

model_name = "model_MLPDis_p2_" + str(epochs) + 'ep_seed' + str(rng_seed) + '.pth'
y_pred_p2 = trainMLPDis(X_train_p2, y_train_p2, climate_train, X_test_p, epochs, model_name, device)

yp_2=pd.DataFrame(y_pred_p2, columns=['Label_lev2_pred'])
yp_2.index = yp_p.index
yp_p = pd.concat([yp_p, yp_2], axis=1, ignore_index=False)

# eval lev 2
crop_idx = (LU22_test_prime['Label_lev1_code'] == 200)
y_test_p2 = LU22_test_prime[crop_idx]['Label_lev2_code'].to_numpy()
f1_Lev2_p2 = f1_score(y_test_p2, y_pred_p2[crop_idx], average='weighted')
print(f'F1 score level 2 (on GT crop only): {f1_Lev2_p2}')

# replace level-2 class values only on pixels classified as 200 in level-1 classification
def replace_values(row):
    if str(row['Label_lev1_pred']).startswith('2'):
        return row['Label_lev2_pred']
    else:
        return row['Label_lev1_pred']
    
yp_p['Label_lev2_pred'] = yp_p.apply(replace_values, axis=1)


#### gap_fill Classification

# keep remaining test points
LU22_test_rem = LU22_test_gapfill[~LU22_test_gapfill['pointid'].isin(set(LU22_test_prime['pointid']))]
yp_g=LU22_test_rem[['pointid','Label_lev1_code','Label_lev2_code']]

# Lev1
climate_train = LU22_train_gapfill['climate'].copy()
X_train_g1= LU22_train_gapfill[f_gapfill.iloc[:,0]] #select prime important features
y_train_g1= LU22_train_gapfill['Label_lev1_code']

X_test_g= LU22_test_rem[f_gapfill.iloc[:,0]] #select prime important features
y_test_g= LU22_test_rem['Label_lev1_code']

model_name = "model_MLPDis_g1_" + str(epochs) + 'ep_seed' + str(rng_seed) + '.pth'
y_pred_g1 = trainMLPDis(X_train_g1, y_train_g1, climate_train, X_test_g, epochs, model_name)

yp_1=pd.DataFrame(y_pred_g1, columns=['Label_lev1_pred'])
yp_1.index = yp_g.index
yp_g = pd.concat([yp_g, yp_1], axis=1, ignore_index=False)  

# eval lev 1
f1_Lev1_g1 = f1_score(y_test_g, y_pred_g1, average='weighted')
print(f'REMAINING DATA')
print(f'F1 score level 1: {f1_Lev1_g1}')

# lev2

LU22_train_gapfill_2=LU22_train_gapfill.copy()
LU22_train_gapfill_2=LU22_train_gapfill_2.loc[LU22_train_gapfill_2['Label_lev1_code'] == 200]

climate_train = LU22_train_gapfill_2['climate'].copy()
X_train_g2=LU22_train_gapfill_2[f_gapfill.iloc[:,0]]
y_train_g2 = LU22_train_gapfill_2['Label_lev2_code']

model_name = "model_MLPDis_g2_" + str(epochs) + 'ep_seed' + str(rng_seed) + '.pth'
y_pred_g2 = trainMLPDis(X_train_g2, y_train_g2, climate_train, X_test_g, epochs, model_name)

yp_2=pd.DataFrame(y_pred_g2, columns=['Label_lev2_pred'])
yp_2.index = yp_g.index
yp_g = pd.concat([yp_g, yp_2], axis=1, ignore_index=False)

yp_g['Label_lev2_pred'] = yp_g.apply(replace_values, axis=1)
# print(f'NaNs in yp_g \n {yp_g.isna().sum()}')

# eval lev 2
crop_idx = (LU22_test_rem['Label_lev1_code'] == 200)
y_test_g2 = LU22_test_rem[crop_idx]['Label_lev2_code'].to_numpy()
f1_Lev2_g2 = f1_score(y_test_g2, y_pred_g2[crop_idx], average='weighted')
print(f'F1 score level 2 (on GT crop only): {f1_Lev2_g2}')


### Final assessment

yp= pd.concat([yp_p, yp_g])

accuracy_Lev1= accuracy_score(yp['Label_lev1_code'], yp['Label_lev1_pred'])
f1_Lev1= f1_score(yp['Label_lev1_code'], yp['Label_lev1_pred'], average="weighted")
kappa_Lev1=cohen_kappa_score(yp['Label_lev1_code'], yp['Label_lev1_pred'])
cf_Lev1=CF(yp['Label_lev1_code'], yp['Label_lev1_pred'])

accuracy_Lev2= accuracy_score(yp['Label_lev2_code'], yp['Label_lev2_pred'])
f1_Lev2= f1_score(yp['Label_lev2_code'], yp['Label_lev2_pred'], average="weighted")
kappa_Lev2=cohen_kappa_score(yp['Label_lev2_code'], yp['Label_lev2_pred'])
cf_Lev2=CF(yp['Label_lev2_code'], yp['Label_lev2_pred'])

print(f'FINAL ASSESSMENT')
print(f'F1 score level 1: {f1_Lev1}')
print(f'F1 score level 2: {f1_Lev2}')
