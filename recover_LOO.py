import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from misc import MLP, MLPDisentanglePos, MLPDisentangleV4, normalizeFeatures, loadData, evaluation
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
import joblib
import sys
import pandas as pd
import xgboost as xgb
from sklearn.svm import SVC

# ################################
# Script main body
rng_seed = 0
climate_regions = ['Alpine', 'Atlantic', 'BlackSea', 'Boreal', 'Continental', 'Mediterranean', 'Pannonian', 'Steppic']
training_batch_size = 128
data_path = '../LU22_final_shared/'
model_type = sys.argv[1] if len(sys.argv) > 1 else "RF" # "MLP" "MLP_Dis_posEnc" "MLP_DisMulti_posEnc" "RF" "XGBoost" "SVM"
model_path = f'./results/LOO/{model_type}/'

normalize_features = False if model_type == "RF" else True
epochs = 500
epochs_str = '_' + str(epochs) + 'ep' if model_type.startswith("MLP") else ''
if model_type in ["RF", "SVM"]:
    extension = '.joblib' 
elif model_type == "XGBoost":
    extension = ".json"
else: 
    extension = ".pth"

for suffix in ['prime', 'gapfill']:
    for pred_level in [1, 2]:
        acc_allLOO, F1_allLOO, acc_allLOO_EMA, F1_allLOO_EMA = [], [], [], []

        for loo_region in climate_regions:

            loo = '_LOO-' + loo_region if loo_region else ''
            config_details = model_type + '_' + suffix + loo + '_Lev' + str(pred_level) + epochs_str + '_seed' + str(rng_seed)
            model_name = "model_" + config_details + extension

            ######## Data preparation
            print('Loading data...')
            train_data, train_label, test_data, test_label, climate_train, _, test_geo_enc = loadData(data_path, suffix, pred_level, loo_region)

            # Normalize data
            if normalize_features:
                train_data, feat_min, feat_max = normalizeFeatures(train_data)
                test_data, _, _ = normalizeFeatures(test_data, feat_min, feat_max)

            n_classes = len(np.unique(train_label))

            # Remove NaN climate regions and count number of regions in training data
            valid_indices = ~pd.isna(climate_train)  # Use pandas to handle NaN
            n_domains = len(np.unique(np.array(climate_train[valid_indices], dtype=str)))

            print(f'train_data shape: {train_data.shape}')
            print(f'train_label shape: {train_label.shape}')
            print(f'test_data shape: {test_data.shape}')
            print(f'test_label shape: {test_label.shape}')

            x_test = torch.tensor(test_data, dtype=torch.float32)
            y_test = torch.tensor(test_label, dtype=torch.int64)

            if "posEnc" in model_type:
                coord_test = torch.tensor(test_geo_enc, dtype=torch.float32)
                test_dataset = TensorDataset(x_test, y_test, coord_test)
            else:
                test_dataset = TensorDataset(x_test, y_test)
            test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1024)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            ### Final assessment
            print(f'Loading model weights (previously trained)...')
            if model_type == "MLP":
                model = MLP(n_classes).to(device)
            elif model_type == "MLP_posEnc":
                model = MLPDisentanglePos(n_classes).to(device)
            elif model_type == "MLP_Dis_posEnc":
                model = MLPDisentanglePos(n_classes).to(device)
            elif model_type == "MLP_DisMulti_posEnc":
                model = MLPDisentanglePos(n_classes,num_domains=n_domains).to(device)
            elif model_type == "MLP_DisMulti":
                model = MLPDisentangleV4(n_classes,num_domains=n_domains).to(device)
            elif model_type in ["RF", "SVM"]:
                model = joblib.load(model_path + model_name)
                pred_test = model.predict(test_data)
            elif model_type == "XGBoost":
                model = xgb.Booster()
                # model = xgb.XGBRegressor() # Scikit-learn interface
                model.load_model(model_path + model_name)
                dtest = xgb.DMatrix(test_data)
                pred_test = model.predict(dtest)
            
            if model_type.startswith('MLP'):
                model.load_state_dict(torch.load(model_path + model_name,map_location=torch.device(device)))
                pred_test, labels_test = evaluation(model, test_dataloader, device)

            acc = accuracy_score(test_label, pred_test)
            kappa=cohen_kappa_score(test_label, pred_test)
            f1 = f1_score(test_label, pred_test, average='weighted')
            f1_perclass = f1_score(test_label, pred_test, average=None)

            F1_allLOO.append(f1)
            acc_allLOO.append(acc)

            ### Final assessment - EMA model
            if model_type.startswith("MLP"):
                print(f'Loading model weights (previously trained)...')
                model.load_state_dict(torch.load(model_path + 'EMA' + model_name,map_location=torch.device(device)))

                pred_test, labels_test = evaluation(model, test_dataloader, device)
                acc = accuracy_score(labels_test, pred_test)
                kappa=cohen_kappa_score(labels_test, pred_test)
                f1 = f1_score(labels_test, pred_test, average='weighted')
                f1_perclass = f1_score(labels_test, pred_test, average=None)

                F1_allLOO_EMA.append(f1)
                acc_allLOO_EMA.append(acc)

        print(f'\n\n\n>>> Scenario: Data {suffix}, Level {pred_level}')
        np.set_printoptions(precision=2)
        print(f'Acc LOO: {np.array2string(np.array(acc_allLOO)*100, separator=", ")}')
        print(f'F1 LOO: {np.array2string(np.array(F1_allLOO)*100, separator=", ")}')

        if model_type.startswith("MLP"):
            print('\n--- EMA model assessment ---')
            print(f'Acc LOO: {np.array2string(np.array(acc_allLOO_EMA)*100, separator=", ")}')
            print(f'F1 LOO: {np.array2string(np.array(F1_allLOO_EMA)*100, separator=", ")}')

        print('\n\n\n')
