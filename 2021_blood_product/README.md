# Personalized surgical transfusion risk prediction using machine learning

## Scripts
1. NSQIP_train.py
    - trains penalized logistic regression, decision tree, random forest, and XGBoost models to predict surgical transfusion
    - usage: `python NSQIP_train.py`
    - prints output to stdout
2. single_var_model.py
    - semi-sklearn compatible wrapper to write .joblib for single variable baseline model using only historical surgery-specific transfusion rates
    - usage: `python single_var_model.py`
3. eval_model.py
    - evaluate model performance on internal and external validation data, perform bootstrapping, and make plots
    - usage: `python eval_model.py`
    - prints output to stdout, also writes .csv files for bootstrap output
4. shap_explain.py
    - usage: `python shap_explain.py`
    - might be best to run in an ipython or jupyter session so SHAP values are available for further manipulation
5. utils.py
    - utility classes and functions needed by other scripts
6. NSQIP_clean.R
    - processes raw NSQIP PUF files


## Folders
- /models/ 
    - .joblib saved models for baseline, LR, DT, XGB (Random Forest is not provided because file size is prohibitively large, but is available on request)
- /result/
    - scripts are hard-coded to drop output files here
