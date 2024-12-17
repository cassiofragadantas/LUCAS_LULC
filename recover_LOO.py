import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from misc import MLP, normalizeFeatures, loadData, evaluation
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score


# ################################
# Script main body
rng_seed = 0
epochs = 300
climate_regions = ['Alpine', 'Atlantic', 'BlackSea', 'Boreal', 'Continental', 'Mediterranean', 'Pannonian', 'Steppic']
training_batch_size = 128
normalize_features = True
data_path = '../LU22_final_shared/'
model_path = './results/LOO/MLP_300/'
model_type = "MLP" # "MLP_Dis_posEnc" "MLP_DisMulti_posEnc"


for suffix in ['prime', 'gapfill']:
    for pred_level in [1, 2]:
        acc_allLOO, F1_allLOO, acc_allLOO_EMA, F1_allLOO_EMA = [], [], [], []

        for loo_region in climate_regions:

            loo = '_LOO-' + loo_region if loo_region else ''
            config_details = model_type + '_' + suffix + loo + '_Lev' + str(pred_level) + '_' + str(epochs) + 'ep_seed' + str(rng_seed)
            model_name = "model_" + config_details + '.pth'

            ######## Data preparation
            print('Loading data...')
            train_data, train_label, test_data, test_label, _,_,_ = loadData(data_path, suffix, pred_level, loo_region)

            # Normalize data
            if normalize_features:
                train_data, feat_min, feat_max = normalizeFeatures(train_data)
                test_data, _, _ = normalizeFeatures(test_data, feat_min, feat_max)

            n_classes = len(np.unique(train_label))

            print(f'train_data shape: {train_data.shape}')
            print(f'train_label shape: {train_label.shape}')
            print(f'test_data shape: {test_data.shape}')
            print(f'test_label shape: {test_label.shape}')

            x_test = torch.tensor(test_data, dtype=torch.float32)
            y_test = torch.tensor(test_label, dtype=torch.int64)

            test_dataset = TensorDataset(x_test, y_test)
            test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1024)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = MLP(n_classes).to(device)

            ### Final assessment
            print(f'Loading model weights (previously trained)...')
            model.load_state_dict(torch.load(model_path + model_name,map_location=torch.device(device)))

            pred_test, labels_test = evaluation(model, test_dataloader, device)
            acc = accuracy_score(labels_test, pred_test)
            kappa=cohen_kappa_score(labels_test, pred_test)
            f1 = f1_score(labels_test, pred_test, average='weighted')
            f1_perclass = f1_score(labels_test, pred_test, average=None)

            F1_allLOO.append(f1)
            acc_allLOO.append(acc)

            ### Final assessment - EMA model
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
        print(f'MLP TEST perf')
        np.set_printoptions(precision=2)
        print(f'Acc LOO: {np.array2string(np.array(acc_allLOO)*100, separator=", ")}')
        print(f'F1 LOO: {np.array2string(np.array(F1_allLOO)*100, separator=", ")}')

        print('\n--- EMA model assessment ---')
        print(f'Acc LOO: {np.array2string(np.array(acc_allLOO_EMA)*100, separator=", ")}')
        print(f'F1 LOO: {np.array2string(np.array(F1_allLOO_EMA)*100, separator=", ")}')

        print('\n\n\n')
