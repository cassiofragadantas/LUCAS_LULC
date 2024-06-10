import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, classification_report 
from misc import normalizeFeatures, loadData
import joblib


# Input arguments
pred_level = int(sys.argv[1]) if len(sys.argv) > 1 else 2
rng_seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

# prime:  all features but cloud free samples only 
# gapfill: all samples but cloud free features only
suffix = sys.argv[3] if len(sys.argv) > 3 else 'prime' #'prime' or 'gapfill'
data_path = '../LU22_final_shared/'
model_name = "model_RF_" + suffix + '_Lev' + str(pred_level) + '_seed' + str(rng_seed) + '.pth'
normalize_features = True

print(f'(Random seed set to {rng_seed})')
# torch.manual_seed(rng_seed)
np.random.seed(rng_seed)

######## Load data
print('Loading data...')
train_data, train_label, test_data, test_label, climate_train = loadData(data_path, suffix, pred_level)

# Normalize data
if normalize_features:
    train_data, feat_min, feat_max = normalizeFeatures(train_data)
    test_data, _, _ = normalizeFeatures(test_data, feat_min, feat_max)

n_classes = len(np.unique(train_label))

print(f'train_data shape: {train_data.shape}')
print(f'train_label shape: {train_label.shape}')
print(f'test_data shape: {test_data.shape}')
print(f'test_label shape: {test_label.shape}')

######## Model training
print('Training model...')
if suffix == 'prime':
    n_estimators=130 if pred_level==1 else 150
else: # 'gapfill'
    n_estimators=110 if pred_level==1 else 115

model = RandomForestClassifier(n_estimators=130, criterion='gini', max_depth= None, min_samples_leaf=2, max_features= 'sqrt' , oob_score=True)
model.fit(train_data, train_label)

y_pred = model.predict(test_data)

# save
joblib.dump(model, model_name)

### Final assessment
acc = accuracy_score(test_label, y_pred)
kappa=cohen_kappa_score(test_label, y_pred)
f1 = f1_score(test_label, y_pred, average='weighted')
f1_perclass = f1_score(test_label, y_pred, average=None)

print(f'>>> Scenario: Data {suffix}, Level {pred_level}')
print(f'RF TEST perf: Acc={acc*100:.2f}, F1={f1*100:.2f}')
np.set_printoptions(precision=2)
print(f'F1 per-class: {f1_perclass*100}')
# print(confusion_matrix(y_pred, test_label))
