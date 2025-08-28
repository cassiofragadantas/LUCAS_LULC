import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
import os
import sys
import numpy as np
import pandas as pd
from tab_transformer_pytorch import FTTransformer # $ pip install tab-transformer-pytorch
from misc import normalizeFeatures, loadData, cumulate_EMA, plot_confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, ConfusionMatrixDisplay
from torch.optim.lr_scheduler import CosineAnnealingLR
from misc import evaluation_categ


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
config_details = "FTTransf_" + suffix + loo + '_Lev' + str(pred_level) + '_' + str(epochs) + 'ep_seed' + str(rng_seed)
model_name = "model_" + config_details + '.pth'

scheduler = True

print(f'(Random seed set to {rng_seed})')
torch.manual_seed(rng_seed)
np.random.seed(rng_seed)

######## Data preparation
print('Loading data...')
train_data, train_label, test_data, test_label, _,_,_,_,_,_ = loadData(data_path, suffix, pred_level, loo_region)

# Normalize data
if normalize_features:
    train_data, feat_min, feat_max = normalizeFeatures(train_data)
    test_data, _, _ = normalizeFeatures(test_data, feat_min, feat_max)
# valid_data =  # Could be part of the traning dataset
# valid_label = 

n_classes = len(np.unique(train_label))

print(f'train_data shape: {train_data.shape}')
print(f'train_label shape: {train_label.shape}')
print(f'test_data shape: {test_data.shape}')
print(f'test_label shape: {test_label.shape}')

x_train = torch.tensor(train_data, dtype=torch.float32)
y_train = torch.tensor(train_label, dtype=torch.int64)

# x_valid = torch.tensor(valid_data, dtype=torch.float32)
# y_valid = torch.tensor(valid_label, dtype=torch.int64)

x_test = torch.tensor(test_data, dtype=torch.float32)
y_test = torch.tensor(test_label, dtype=torch.int64)

train_dataset = TensorDataset(x_train, y_train)
# valid_dataset = TensorDataset(x_valid, y_valid)
test_dataset = TensorDataset(x_test, y_test)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=training_batch_size)
# valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=1024)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1024)

######## Model training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Model
model = FTTransformer(
    categories = (),                    # tuple containing the number of unique values within each category
    num_continuous = x_train.shape[1],  # number of continuous values
    dim = 32,                           # dimension, paper set at 32
    dim_out = n_classes,                # output dimension
    depth = 6,                          # depth, paper recommended 6
    heads = 8,                          # heads, paper recommends 8
    attn_dropout = 0.1,                 # post-attention dropout
    ff_dropout = 0.1                    # feed forward dropout
).to(device)
print(model)

if os.path.isfile(model_name):
    print(f'Loading model weights (previously trained)...')
    model.load_state_dict(torch.load(model_name,map_location=torch.device(device)))
else:
    print(f'Training model on {device}...')

    learning_rate = 0.0001
    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

    if scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

    ema_weights = None

    for epoch in range(epochs):
        start = time.time()
        model.train()
        tot_loss, den = 0., 0.
        
        for x_batch, y_batch in train_dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            x_categ_empty = torch.empty(x_batch.size(0), 0, dtype=torch.long, device=x_batch.device)

            optimizer.zero_grad()
            pred = model(x_categ_empty, x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward() # backward pass: backpropagate the prediction loss
            optimizer.step() # gradient descent: adjust the parameters by the gradients collected in the backward pass
            tot_loss+= loss.cpu().detach().numpy()
            den+=1.

        if scheduler:
            scheduler.step()

        end = time.time()

        # Evaluation
        with torch.no_grad():
            if use_valid:
                pred_valid, labels_valid = evaluation_categ(model, valid_dataloader, device)
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
                # pred_test, labels_test = evaluation_categ(model, test_dataloader, device)
                # f1_ema = f1_score(labels_test, pred_test, average="weighted")
                # model.load_state_dict(current_state_dict)
            ####################### EMA #####################################

            if eval_test:
                pred_test, labels_test = evaluation_categ(model, test_dataloader, device)
                # acc = accuracy_score(labels_test, pred_test)
                f1 = f1_score(labels_test, pred_test, average="weighted")
                print("Epoch %d (%.2fs): train loss %.4f. F1 on TEST %.2f"%(epoch, (end-start), tot_loss/den, 100*f1))
                #print(confusion_matrix(labels_test, pred_test))        
            else:
                print("Epoch %d (%.2fs): train loss %.4f"%(epoch, (end-start), tot_loss/den))

### Model parameter count
total_params = sum(p.numel() for p in model.parameters())
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal Model Params: {total_params}")
print(f"Total Model Trainable Params: {total_trainable_params}\n")

### Final assessment
start_time = time.time()
pred_test, labels_test = evaluation_categ(model, test_dataloader, device)
execution_time = time.time() - start_time
print(f"Inference time: {execution_time:.6f} seconds")
acc = accuracy_score(labels_test, pred_test)
kappa=cohen_kappa_score(labels_test, pred_test)
f1 = f1_score(labels_test, pred_test, average='weighted')
f1_perclass = f1_score(labels_test, pred_test, average=None)

print(f'>>> Scenario: Data {suffix}, Level {pred_level}')
print(f'FTTransf TEST perf: Acc={acc*100:.2f}, F1={f1*100:.2f}')
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

pred_test, labels_test = evaluation_categ(model, test_dataloader, device)
acc = accuracy_score(labels_test, pred_test)
kappa=cohen_kappa_score(labels_test, pred_test)
f1 = f1_score(labels_test, pred_test, average='weighted')
f1_perclass = f1_score(labels_test, pred_test, average=None)

print('\n--- EMA model assessment ---')
print(f'FTTransf TEST perf: Acc={acc*100:.2f}, F1={f1*100:.2f}')
np.set_printoptions(precision=2)
print(f'F1 per-class: {f1_perclass*100}')

