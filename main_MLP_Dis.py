import torch
import torch.nn as nn
import sys
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import os
from misc import MLPDisentangleV4, SupervisedContrastiveLoss, normalizeFeatures, loadData, cumulate_EMA, evaluation
import time
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, cohen_kappa_score
import torch.nn.functional as F
#import warnings
#warnings.filterwarnings('ignore')
#torch.set_default_dtype(torch.float16)


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

# C + 2 classes: C for domain invariant + source domain spec + target domain spec
def sup_contra_Cplus2_classes(emb, ohe_label, ohe_dom, scl, epoch):
    norm_emb = nn.functional.normalize(emb)
    C = ohe_label.max() + 1
    new_combined_label = [v1 if v2==2 else C+v2 for v1, v2 in zip(ohe_label, ohe_dom)]
    new_combined_label = torch.tensor(np.array(new_combined_label), dtype=torch.int64)
    return scl(norm_emb, new_combined_label, epoch=epoch)

# ################################
# Script main body

# Input arguments
pred_level = int(sys.argv[1]) if len(sys.argv) > 1 else 2
rng_seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

# prime:  all features but cloud free samples only 
# gapfill: all samples but cloud free features only
suffix = sys.argv[3] if len(sys.argv) > 3 else 'prime' #'prime' or 'gapfill'
epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 100
training_batch_size = 128
normalize_features = True
use_valid = False # use validation dataset
momentum_ema = .95
data_path = '../LU22_final_shared/'
model_name = "model_MLP_Dis_climate_" + suffix + '_Lev' + str(pred_level) + '_' + str(epochs) + 'ep_seed' + str(rng_seed) + '.pth'

print(f'(Random seed set to {rng_seed})')
torch.manual_seed(rng_seed)
np.random.seed(rng_seed)

######## Data preparation
print('Loading data...')
train_data, train_label, test_data, test_label, climate_train = loadData(data_path, suffix, pred_level)

# Normalize data
if normalize_features:
    train_data, feat_min, feat_max = normalizeFeatures(train_data)
    test_data, _, _ = normalizeFeatures(test_data, feat_min, feat_max)

# valid_data =  # Could be part of the traning dataset
# valid_label = 

# Separate by climate region
idx_med = (climate_train == 'Mediterranean')
train_target_data = train_data[idx_med]
train_target_label = train_label[idx_med]

train_source_data = train_data[~idx_med]
train_source_label = train_label[~idx_med]

train_data = np.concatenate([train_target_data, train_source_data],axis=0)
train_label = np.concatenate([train_target_label, train_source_label],axis=0)
train_domain_label = np.concatenate([np.zeros(train_target_label.shape[0]), np.ones(train_source_label.shape[0])], axis=0)

n_classes = len(np.unique(train_label))

print(f'train_data shape: {train_data.shape}')
print(f'train_label shape: {train_label.shape}')
print(f'\t>> source {train_source_label.shape}, taget {train_target_label.shape}')
print(f'test_data shape: {test_data.shape}')
print(f'test_label shape: {test_label.shape}')

x_train = torch.tensor(train_data, dtype=torch.float32)
y_train = torch.tensor(train_label, dtype=torch.int64)
dom_train = torch.tensor(train_domain_label, dtype=torch.int64)

# x_valid = torch.tensor(valid_data, dtype=torch.float32)
# y_valid = torch.tensor(valid_label, dtype=torch.int64)

x_test = torch.tensor(test_data, dtype=torch.float32)
y_test = torch.tensor(test_label, dtype=torch.int64)

train_dataset = TensorDataset(x_train, y_train, dom_train)
# valid_dataset = TensorDataset(x_valid, y_valid)
test_dataset = TensorDataset(x_test, y_test)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=training_batch_size)
# valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=1024)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1024)

######## Model training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Running on {device}')

# Model
# 1) Random Forest (theirs)
# model = RandomForestClassifier(n_estimators=130, criterion='gini', max_depth= None, min_samples_leaf=2, max_features= 'sqrt' , oob_score=True)
# model.fit(x_train, y_train)
# y_pred = model.predict(X_test)
# 2) Ours
model = MLPDisentangleV4(n_classes).to(device)
# model = TempCNNDisentangleV4(n_classes).to(device)


learning_rate = 0.0001
loss_fn = nn.CrossEntropyLoss()
scl = SupervisedContrastiveLoss()

optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

ema_weights = None

for epoch in range(epochs):
    start = time.time()
    model.train()
    tot_loss = 0.0
    domain_loss = 0.0
    contra_tot_loss = 0.0
    den = 0

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

    # Evaluation
    with torch.no_grad():
        if use_valid:
            pred_valid, labels_valid = evaluation(model, valid_dataloader, device)
            f1_val = f1_score(labels_valid, pred_valid, average="weighted")
            
            eval_test = (f1_val > valid_f1)
            if f1_val > valid_f1:
                torch.save(model.state_dict(), model_name)
                valid_f1 = f1_val
        else:
            torch.save(model.state_dict(), model_name)
            eval_test = (epoch%10 == 0)


        ####################### EMA #####################################
        if epoch >= epochs/2:        
            ema_weights = cumulate_EMA(model, ema_weights, momentum_ema)
            # current_state_dict = model.state_dict()        
            # model.load_state_dict(ema_weights)
            # pred_test, labels_test = evaluation(model, test_dataloader, device)
            # f1_ema = f1_score(labels_test, pred_test, average="weighted")
            # model.load_state_dict(current_state_dict)
        ####################### EMA #####################################

        if eval_test:
            pred_test, labels_test = evaluation(model, test_dataloader, device)
            # acc = accuracy_score(labels_test, pred_test)
            f1 = f1_score(labels_test, pred_test, average="weighted")
            print("Epoch %d (%.2fs): train loss %.4f, contrastive %.4f. F1 on TEST %.2f"%(epoch, (end-start), tot_loss/den, contra_tot_loss/den, 100*f1))
            #print(confusion_matrix(labels_test, pred_test))        
        else:
            print("Epoch %d (%.2fs): train loss %.4f, contrastive %.4f"%(epoch, (end-start), tot_loss/den, contra_tot_loss/den))
    
        sys.stdout.flush()



### Final assessment
pred_test, labels_test = evaluation(model, test_dataloader, device)
acc = accuracy_score(labels_test, pred_test)
kappa=cohen_kappa_score(labels_test, pred_test)
f1 = f1_score(labels_test, pred_test, average='weighted')
f1_perclass = f1_score(labels_test, pred_test, average=None)

print(f'>>> Scenario: Data {suffix}, Level {pred_level}')
print(f'MLP+dis TEST perf: Acc={acc*100:.2f}, F1={f1*100:.2f}')
np.set_printoptions(precision=2)
print(f'F1 per-class: {f1_perclass*100}')
# print(confusion_matrix(labels_test, pred_test))

### Final assessment - EMA model
model.load_state_dict(ema_weights)
torch.save(model.state_dict(), 'EMA' + model_name)

pred_test, labels_test = evaluation(model, test_dataloader, device)
acc = accuracy_score(labels_test, pred_test)
kappa=cohen_kappa_score(labels_test, pred_test)
f1 = f1_score(labels_test, pred_test, average='weighted')
f1_perclass = f1_score(labels_test, pred_test, average=None)

print('\n--- EMA model assessment ---')
print(f'MLP+dis TEST perf: Acc={acc*100:.2f}, F1={f1*100:.2f}')
np.set_printoptions(precision=2)
print(f'F1 per-class: {f1_perclass*100}')

