import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
import os
import sys
import numpy as np
import pandas as pd
from misc import MLPDisentanglePosFixed, SupervisedContrastiveLoss, normalizeFeatures, loadData, cumulate_EMA, evaluation
from misc import sim_dist_specifc_loss_spc, sup_contra_Cplus2_classes, plot_confusion_matrix
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, cohen_kappa_score, ConfusionMatrixDisplay
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
#import warnings
#warnings.filterwarnings('ignore')
#torch.set_default_dtype(torch.float16)


# ################################
# Script main body

# Input arguments
pred_level = int(sys.argv[1]) if len(sys.argv) > 1 else 2
rng_seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

# prime:  all features but cloud free samples only 
# gapfill: all samples but cloud free features only
suffix = sys.argv[3] if len(sys.argv) > 3 else 'prime' #'prime' or 'gapfill'
epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 300
loo_region = sys.argv[5] if len(sys.argv) > 5 else None # Held-out climatic region (see options below). If 'None', the usual train-test split is used
# Regions: ['Alpine' 'Atlantic' 'BlackSea' 'Boreal' 'Continental' 'Mediterranean' 'Pannonian' 'Steppic']
training_batch_size = 128
normalize_features = True
use_valid = False # use validation dataset
momentum_ema = .95
data_path = '../LU22_final_shared/'
loo = '_LOO-' + loo_region if loo_region else ''
config_details = "MLP_DisMulti_posEncFixed_" + suffix + loo + '_Lev' + str(pred_level) + '_' + str(epochs) + 'ep_seed' + str(rng_seed)
model_name = "model_" + config_details + '.pth'

scheduler = True

print(f'(Random seed set to {rng_seed})')
torch.manual_seed(rng_seed)
np.random.seed(rng_seed)

######## Data preparation
print('Loading data...')
train_data, train_label, test_data, test_label, \
    climate_train, _, train_geo_enc, test_geo_enc, _, _ = loadData(data_path, suffix, pred_level, loo_region)

# Normalize data
if normalize_features:
    train_data, feat_min, feat_max = normalizeFeatures(train_data)
    test_data, _, _ = normalizeFeatures(test_data, feat_min, feat_max)

# valid_data =  # Could be part of the traning dataset
# valid_label = 

# Filter out NaN values from climate_train
valid_indices = ~pd.isna(climate_train)  # Use pandas to handle NaN
print(f'Removing {pd.isna(climate_train).sum()} samples with NaN climate regions')
climate_train = climate_train[valid_indices]
train_data = train_data[valid_indices]
train_label = train_label[valid_indices]
train_geo_enc = train_geo_enc[valid_indices]

# Separate by climate region
climate_train = np.array(climate_train, dtype=str)
unique_climates = np.unique(climate_train)

# Lists to hold separated datasets
train_data_list = []
train_label_list = []
train_geo_enc_list = []
train_domain_label_list = []

# Populate the lists
for domain_id, climate in enumerate(unique_climates):
    idx_climate = (climate_train == climate)  # Get indexes for the current climate
    # Append the target (current climate) data
    train_data_list.append(train_data[idx_climate])
    train_label_list.append(train_label[idx_climate])
    train_geo_enc_list.append(train_geo_enc[idx_climate])
    # Create domain labels corresponding to the current climate
    train_domain_label_list.append(np.full(idx_climate.sum(), domain_id))    

# Concatenate all the target and source data
train_data = np.concatenate(train_data_list, axis=0)
train_label = np.concatenate(train_label_list, axis=0)
train_geo_enc = np.concatenate(train_geo_enc_list, axis=0)
train_domain_label = np.concatenate(train_domain_label_list, axis=0)

n_classes = len(np.unique(train_label))
n_domains = len(unique_climates)

print(f'train_data shape: {train_data.shape}')
print(f'train_geo_enc shape: {train_geo_enc.shape}')
print(f'train_label shape: {train_label.shape}')
print(f'test_data shape: {test_data.shape}')
print(f'test_label shape: {test_label.shape}')

x_train = torch.tensor(train_data, dtype=torch.float32)
y_train = torch.tensor(train_label, dtype=torch.int64)
dom_train = torch.tensor(train_domain_label, dtype=torch.int64)
coord_train = torch.tensor(train_geo_enc, dtype=torch.float32)

# x_valid = torch.tensor(valid_data, dtype=torch.float32)
# y_valid = torch.tensor(valid_label, dtype=torch.int64)

x_test = torch.tensor(test_data, dtype=torch.float32)
y_test = torch.tensor(test_label, dtype=torch.int64)
coord_test = torch.tensor(test_geo_enc, dtype=torch.float32)
print(f'x_test.shape {x_test.shape}')
print(f'coord_test.shape {coord_test.shape}')

train_dataset = TensorDataset(x_train, y_train, dom_train, coord_train)
# valid_dataset = TensorDataset(x_valid, y_valid)
test_dataset = TensorDataset(x_test, y_test, coord_test)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=training_batch_size)
# valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=1024)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1024)

######## Model training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Model
# 1) Random Forest (theirs)
# model = RandomForestClassifier(n_estimators=130, criterion='gini', max_depth= None, min_samples_leaf=2, max_features= 'sqrt' , oob_score=True)
# model.fit(x_train, y_train)
# y_pred = model.predict(X_test)
# 2) Ours
model = MLPDisentanglePosFixed(n_classes,num_domains=n_domains).to(device)
# model = TempCNNDisentangleV4(n_classes).to(device)

if os.path.isfile(model_name):
    print(f'Loading model weights (previously trained)...')
    model.load_state_dict(torch.load(model_name,map_location=torch.device(device)))
else:
    print(f'Training model on {device}...')

    learning_rate = 0.0001
    loss_fn = nn.CrossEntropyLoss()
    scl = SupervisedContrastiveLoss()

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

    if scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0) 

    ema_weights = None

    for epoch in range(epochs):
        start = time.time()
        model.train()
        tot_loss = 0.0
        domain_loss = 0.0
        contra_tot_loss = 0.0
        den = 0

        for x_batch, y_batch, dom_batch, coord_batch in train_dataloader:
            if x_batch.shape[0] != training_batch_size:
                continue

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            dom_batch = dom_batch.to(device)
            coord_batch = coord_batch.to(device)
            optimizer.zero_grad()
            pred, inv_emb, spec_emb_d, spec_d_pred, inv_emb_n1, spec_emb_n1, inv_fc_feat, spec_fc_feat = model(x_batch, coord_batch)

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
            #DOMAIN LABEL FOR INV EMBEDDING IS n_domains
            inv_dom_labels = np.ones_like(spec_dc_dom_labels) * n_domains

            dom_mix_labels = np.concatenate([inv_dom_labels, spec_dc_dom_labels],axis=0)
            
            joint_embedding = torch.concat([inv_emb, spec_emb_d])
            mixdl_loss_supContraLoss = sim_dist_specifc_loss_spc(joint_embedding, y_mix_labels, dom_mix_labels, scl, epoch)
            
            joint_embedding_n1 = torch.concat([inv_emb_n1, spec_emb_n1])
            mixdl_loss_supContraLoss_n1 = sim_dist_specifc_loss_spc(joint_embedding_n1, y_mix_labels, dom_mix_labels, scl, epoch)

            joint_embedding_fc_feat = torch.concat([inv_fc_feat, spec_fc_feat])
            # mixdl_loss_supContraLoss_fc = sim_dist_specifc_loss_spc(joint_embedding_fc_feat, y_mix_labels, dom_mix_labels, scl, epoch)
            mixdl_loss_supContraLoss_fc = sup_contra_Cplus2_classes(joint_embedding_fc_feat, y_mix_labels, dom_mix_labels, scl, epoch)
            
            contra_loss = mixdl_loss_supContraLoss_fc #+ mixdl_loss_supContraLoss_n1 + mixdl_loss_supContraLoss 

            ####################################

            loss = loss_fn(pred, y_batch) + contra_loss + loss_ce_spec_dom

            loss.backward() # backward pass: backpropagate the prediction loss
            optimizer.step() # gradient descent: adjust the parameters by the gradients collected in the backward pass
            tot_loss+= loss.cpu().detach().numpy()
            contra_tot_loss+= contra_loss.cpu().detach().numpy()
            den+=1.

        if scheduler:
            scheduler.step()

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

# Confusion matrix
conf_matrix = confusion_matrix(labels_test, pred_test)
# save to csv
cm_filename = "confusion_matrix_" + config_details
df_conf_matrix = pd.DataFrame(conf_matrix)
df_conf_matrix.to_csv(cm_filename + '.csv', index=False)
# Plot
text = True if pred_level==1 else False # Display values on each cell in confusion matrix
cm_title = f'Confusion Matrix ({suffix}, level {pred_level})'
plot_confusion_matrix(conf_matrix, title=cm_title, filename = cm_filename, text=text)
# Plot normalized per-row
conf_matrix = confusion_matrix(labels_test, pred_test, normalize='true')
plot_confusion_matrix(conf_matrix, title=cm_title, filename = cm_filename + '_norm', normalized=True, text=text)

### Final assessment - EMA model
if os.path.isfile('EMA' + model_name):
    print(f'Loading model weights (previously trained)...')
    model.load_state_dict(torch.load('EMA' + model_name,map_location=torch.device(device)))
else:
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

