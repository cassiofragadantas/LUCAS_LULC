import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, ConfusionMatrixDisplay
from misc import normalizeFeatures, loadData, plot_confusion_matrix
import time


# Input arguments
pred_level = int(sys.argv[1]) if len(sys.argv) > 1 else 2
rng_seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

# prime:  all features but cloud free samples only 
# gapfill: all samples but cloud free features only
suffix = sys.argv[3] if len(sys.argv) > 3 else 'prime' #'prime' or 'gapfill'
epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 100
normalize_features = True
data_path = '../LU22_final_shared/'
config_details = "XGBoost_" + suffix + '_Lev' + str(pred_level) + '_seed' + str(rng_seed)
model_name = "model_" + config_details + '.json'

print(f'(Random seed set to {rng_seed})')
# torch.manual_seed(rng_seed)
np.random.seed(rng_seed)

######## Load data
print('Loading data...')
train_data, train_label, test_data, test_label, _,_,_ = loadData(data_path, suffix, pred_level)

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
if os.path.isfile(model_name):
    print(f'Loading model weights (previously trained)...')
    model = xgb.Booster()
    # model = xgb.XGBRegressor() # Scikit-learn interface
    model.load_model(model_name)
else:
    print('Training model...')
    start_time = time.time()

    # See default parameters here: https://xgboost.readthedocs.io/en/stable/parameter.html
    params = {
        "objective": "multi:softmax",  # Specify multiclass classification
        "num_class": n_classes,        # Number of classes (Iris has 3 species)
        "eval_metric": "mlogloss",     # Evaluation metric
        "seed": rng_seed,              # Random seed
    }    

    dtrain = xgb.DMatrix(train_data, label=train_label)
    model = xgb.train(params, dtrain, num_boost_round=epochs) # device = 'cpu'

    # Scikit-learn interface
    # model = xgb.XGBRegressor() # device = 'cpu'
    # model.fit(train_data, train_label)
    
    execution_time = time.time() - start_time
    print(f"Training time: {execution_time:.6f} seconds")

    # save
    model.save_model(model_name)

### Inference
start_time = time.time()
dtest = xgb.DMatrix(test_data)
y_pred = model.predict(dtest)
# y_pred = model.predict(test_data) # Scikit-learn interface
execution_time = time.time() - start_time
print(f"Inference time: {execution_time:.6f} seconds")


### Final assessment
acc = accuracy_score(test_label, y_pred)
kappa=cohen_kappa_score(test_label, y_pred)
f1 = f1_score(test_label, y_pred, average='weighted')
f1_perclass = f1_score(test_label, y_pred, average=None)

print(f'>>> Scenario: Data {suffix}, Level {pred_level}')
print(f'XGBoost TEST perf: Acc={acc*100:.2f}, F1={f1*100:.2f}')
np.set_printoptions(precision=2)
print(f'F1 per-class: {f1_perclass*100}')

# Confusion matrix
conf_matrix = confusion_matrix(test_label, y_pred)
# save to csv
cm_filename = "confusion_matrix_" + config_details
df_conf_matrix = pd.DataFrame(conf_matrix)
df_conf_matrix.to_csv(cm_filename + '.csv', index=False)
# Plot
text = True if pred_level==1 else False # Display values on each cell in confusion matrix
cm_title = f'Confusion Matrix ({suffix}, level {pred_level})'
plot_confusion_matrix(conf_matrix, title=cm_title, filename = cm_filename, text=text)
# Plot normalized per-row
conf_matrix = confusion_matrix(test_label, y_pred, normalize='true')
plot_confusion_matrix(conf_matrix, title=cm_title, filename = cm_filename + '_norm', normalized=True, text=text)
