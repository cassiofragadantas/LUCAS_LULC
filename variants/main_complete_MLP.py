import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
from tqdm import tqdm
import sys
import numpy as np
import pandas as pd
from os.path import join
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, cohen_kappa_score, classification_report 
from misc import MLP, normalizeFeatures


data_path = '../LU22_final_shared/'
epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 300
rng_seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'


######## Training loop
def trainMLP(train_data, train_label, test_data, epochs,
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

    x_train = torch.tensor(train_data, dtype=torch.float32)
    y_train = torch.tensor(train_label, dtype=torch.int64)

    train_dataset = TensorDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=training_batch_size)

    print(f'Training model on {device}...')

    n_classes = len(np.unique(train_label))
    model = MLP(n_classes)

    learning_rate = 0.0001
    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(epochs)):
        start = time.time()
        model.train()
        tot_loss, den = 0., 0.
        
        for x_batch, y_batch in train_dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(x_batch)[0]
            loss = loss_fn(pred, y_batch)
            loss.backward() # backward pass: backpropagate the prediction loss
            optimizer.step() # gradient descent: adjust the parameters by the gradients collected in the backward pass
            tot_loss+= loss.cpu().detach().numpy()
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
X_train_p1= LU22_train_prime[f_prime.iloc[:,0]] #select prime important features
y_train_p1= LU22_train_prime['Label_lev1_code']

X_test_p= LU22_test_prime[f_prime.iloc[:,0]] #select prime important features
y_test_p= LU22_test_prime['Label_lev1_code']

model_name = "model_MLP_p1_" + str(epochs) + 'ep_seed' + str(rng_seed) + '.pth'
y_pred_p1 = trainMLP(X_train_p1, y_train_p1, X_test_p, epochs, model_name, device)

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

X_train_p2=LU22_train_prime_2[f_prime.iloc[:,0]]
y_train_p2 = LU22_train_prime_2['Label_lev2_code']

model_name = "model_MLP_p2_" + str(epochs) + 'ep_seed' + str(rng_seed) + '.pth'
y_pred_p2 = trainMLP(X_train_p2, y_train_p2, X_test_p, epochs, model_name, device)

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
X_train_g1= LU22_train_gapfill[f_gapfill.iloc[:,0]] #select prime important features
y_train_g1= LU22_train_gapfill['Label_lev1_code']

X_test_g= LU22_test_rem[f_gapfill.iloc[:,0]] #select prime important features
y_test_g= LU22_test_rem['Label_lev1_code']


model_name = "model_MLP_g1_" + str(epochs) + 'ep_seed' + str(rng_seed) + '.pth'
y_pred_g1 = trainMLP(X_train_g1, y_train_g1, X_test_g, epochs, model_name)

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

X_train_g2=LU22_train_gapfill_2[f_gapfill.iloc[:,0]]
y_train_g2 = LU22_train_gapfill_2['Label_lev2_code']

model_name = "model_MLP_g2_" + str(epochs) + 'ep_seed' + str(rng_seed) + '.pth'
y_pred_g2 = trainMLP(X_train_g2, y_train_g2, X_test_g, epochs, model_name)

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
