# LUCAS_LULC
LULC classification on LUCAS data using ecoregions information

## First conclusions:

See results summary [here](https://docs.google.com/spreadsheets/d/168OG9ZMvTzcGDUVP6YrZDFVrUWTWfii2uJQ8oO_8eE4/edit?usp=sharing)

- MLP improves upon Random Forest results.
- More advanced time series classifiers (like 1D-CNN and InceptionTime approaches) could not be leveraged since the features are not always temporal.
- Exploiting ecoregions information with a feature disentanglement approach did not improve significantly upon the MLP results.
  - Slight improvement on level 1 and slight degradation in level 2.
  - Hypothesis: the disentanglement approach works better when the defined domains are fully distinct, whereas here the ecoregions transition may be smooth (even if a firm frontier is defined).

## Possible improvement directions:

- Provide coordinates information (latitude, longitude) besides the ecoregion label information for the MLP model


## Files description

Main scripts:

- `main_RF.py`: experiments on a chosen nomenclature level (1 or 2) and a chosen data type ('prime' or 'gapfill') with a RandomForest classifier. Launching example for level 1, random seed 0 and prime data: `$python main_RF.py 1 0 prime` (or use the launch_RF.sh file)
- `main_MLP.py`: same as main_RF but with an MLP (Multi-Layer Perceptron) classifier. Launching example with the additional (last) input argument being the number of training epochs: `$python main_MLP.py 1 0 prime 300`
- `main_MLP_Dis.py`: MLP classifier + feature disentanglement approach. Input arguments are the same as in `main_MLP.py`.

Shell scripts (launching experiments):

- `launch_RF.sh`, `launch_MLP.sh`, `launch_MLP_Dis.sh`: launches all variations (levels 1 and 2 for prime and gapfill data) with the corresponding classifier.

Other:

- `misc.py`: contain some supporting functions including models definition.

P.S.: The random seed currently does not affect data splitting (the same fixed split is being used).

## TODO

- Run on different train-test splits (take average and std)
- Run the original two-stage full pipeline for MLP and compare performance.
