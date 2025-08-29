# TabPFN repo: https://github.com/PriorLabs/TabPFN
import os
import sys
import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier, save_fitted_tabpfn_model, load_fitted_tabpfn_model # pip install tabpfn 
from tabpfn_extensions.many_class import ManyClassClassifier # pip install "tabpfn-extensions[all] @ git+https://github.com/PriorLabs/tabpfn-extensions.git"
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, ConfusionMatrixDisplay
from misc import normalizeFeatures, loadData, plot_confusion_matrix
import time
from sklearn.model_selection import train_test_split


# Input arguments
pred_level = int(sys.argv[1]) if len(sys.argv) > 1 else 2
rng_seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

# prime:  all features but cloud free samples only 
# gapfill: all samples but cloud free features only
suffix = sys.argv[3] if len(sys.argv) > 3 else 'prime' #'prime' or 'gapfill'
loo_region = sys.argv[4] if len(sys.argv) > 4 else None # Held-out climatic region (see options below). If 'None', the usual train-test split is used
# Regions: ['Alpine' 'Atlantic' 'BlackSea' 'Boreal' 'Continental' 'Mediterranean' 'Pannonian' 'Steppic']
data_path = '../LU22_final_shared/'
loo = '_LOO-' + loo_region if loo_region else ''
config_details = "TabPFN_" + suffix + loo + '_Lev' + str(pred_level) + '_seed' + str(rng_seed)
model_name = "model_" + config_details + '.tabpfn_fit'
normalize_features = True

print(f'(Random seed set to {rng_seed})')
# torch.manual_seed(rng_seed)
np.random.seed(rng_seed)

######## Load data
print('Loading data...')
train_data, train_label, test_data, test_label, _,_,_,_,_,_ = loadData(data_path, suffix, pred_level, loo_region)

# Normalize data
if normalize_features:
    train_data, feat_min, feat_max = normalizeFeatures(train_data)
    test_data, _, _ = normalizeFeatures(test_data, feat_min, feat_max)

n_classes = len(np.unique(train_label))

# Subsampling datasets
if loo_region:
    n_samples = 50000
    if train_data.shape[0] > n_samples:
        train_data, _, train_label, _ = train_test_split(
            train_data,
            train_label,
            train_size= n_samples,
            stratify=train_label,
            random_state=42
        )
    # if test_data.shape[0] > n_samples + n_classes:        
    #     test_data, _, test_label, _ = train_test_split(
    #         test_data,
    #         test_label,
    #         train_size= n_samples,
    #         stratify=test_label,
    #         random_state=42
    #     )

print(f'train_data shape: {train_data.shape}')
print(f'train_label shape: {train_label.shape}')
print(f'test_data shape: {test_data.shape}')
print(f'test_label shape: {test_label.shape}')

######## Model training
if os.path.isfile(model_name):
    print(f'Loading model weights (previously trained)...')
    model = load_fitted_tabpfn_model(model_name, device="cuda")  # device="cpu"
else:
    print('Training model...')
    start_time = time.time()

    # Create a TabPFN base classifier
    if n_classes > 10:
        # TabPFN's default class limit is often 10 for the public model.
        base_clf = TabPFNClassifier(ignore_pretraining_limits=True) #device='cuda', N_ensemble_configurations=4
        model = ManyClassClassifier(
            estimator=base_clf,
            alphabet_size=10
        )
        print("Many class classifier")
    else:
        model = TabPFNClassifier(ignore_pretraining_limits=True)

    model.fit(train_data, train_label)
    
    if n_classes > 10:
        print(f"codebook: {model.code_book_}")
        print(f"codebook stats: {model.codebook_stats_}")
    
    execution_time = time.time() - start_time
    print(f"Training time: {execution_time:.6f} seconds")

    # save
    if pred_level == 1:
        save_fitted_tabpfn_model(model, model_name)

### Model parameter count
if pred_level==1:
    total_params = sum(p.numel() for p in model.model_.parameters())
    total_trainable_params = sum(p.numel() for p in model.model_.parameters() if p.requires_grad)
    print(f"\nTotal Model Params: {total_params}")
    print(f"Total Model Trainable Params: {total_trainable_params}\n")

### Inference
start_time = time.time()
# y_pred = model.predict(test_data)
# Inference per batch
batch_size = 1000 if pred_level==1 else 5000 # test_data.shape[0]
n_test = test_data.shape[0]
y_pred = []
for start in range(0, n_test, batch_size):
    end = min(start + batch_size, n_test)
    print(f'Inference until sample {end} out of {n_test}')    
    y_pred.append(model.predict(test_data[start:end]))
    execution_time = time.time() - start_time
    print(f"Inference time (until now): {execution_time:.6f} seconds")    
y_pred = np.concatenate(y_pred, axis=0)

### Final assessment
acc = accuracy_score(test_label, y_pred)
kappa=cohen_kappa_score(test_label, y_pred)
f1 = f1_score(test_label, y_pred, average='weighted')
f1_perclass = f1_score(test_label, y_pred, average=None)

print(f'>>> Scenario: Data {suffix}, Level {pred_level}')
print(f'TabPFN TEST perf: Acc={acc*100:.2f}, F1={f1*100:.2f}')
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
