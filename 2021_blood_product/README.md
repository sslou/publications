# Personalized surgical transfusion risk prediction using machine learning to guide preoperative type and screen orders

Code for the models described in [*Anesthesiology*, 2022](https://doi.org/10.1097/ALN.0000000000004139).

## Scripts
1. `NSQIP_train.py`
    - trains penalized logistic regression, decision tree, random forest, and XGBoost models to predict surgical transfusion
    - usage: `python NSQIP_train.py`
    - prints output to stdout
2. `single_var_model.py`
    - semi-sklearn compatible wrapper to write .joblib for single variable baseline model using only historical surgery-specific transfusion rates
    - usage: `python single_var_model.py`
3. `eval_model.py`
    - evaluate model performance on internal and external validation data, perform bootstrapping, and make plots
    - usage: `python eval_model.py`
    - prints output to stdout, also writes .csv files for bootstrap output
4. `shap_explain.py`
    - usage: `python shap_explain.py`
    - example data for a fake patient is included; this script outputs model predicted transfusion risk and illustrates model explanation for this fake patient (see result folder). see also jupyter notebook below.
5. `utils.py`
    - utility classes and functions needed by other scripts
6. `NSQIP_clean.R`
    - processes raw NSQIP PUF files to format that `NSQIP_train.py` can ingest


## Notebooks
[jupyter notebook](https://colab.research.google.com/drive/1PavgJqsxjkRvQ6-2psj-crBCV8gmJZzk?usp=sharing) for making and visualizing model predictions on new patient data, running on Google Colab.


## Folders
- /models/ 
    - `*.joblib` saved models for baseline, LR, DT, XGB (Random Forest is not provided because file size is prohibitively large, but is available on request). Requires sklearn v0.22.1 to load.
- /result/
    - scripts are hard-coded to drop output files here


## Python environment
- python: 3.7.6 
- GCC: 7.3.0
- scipy: 1.4.1
- numpy: 1.18.1
- matplotlib: 3.1.3
- pandas: 1.0.1
- sklearn: 0.22.1
- XGBoost: 1.2.0
- SHAP 0.37.0


*Disclaimer: Please use at your own risk. These models provide estimates of transfusion risk based on certain variables, and there may be other factors not included that can either increase or decrease transfusion risk. These estimates are not a guarantee of results. Transfusion may happen even if the risk is low. This information is not intended to replace the advice of a doctor or healthcare provider about diagnosis, treatment, or potential outcomes.*
