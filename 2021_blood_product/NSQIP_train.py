import sys, joblib
import scipy, sklearn, matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as metrics
from sklearn.calibration import calibration_curve
from utils import ProcessData


def preprocess(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,
                                                      random_state=1, stratify=y)
    
    # median impute missing values
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    imp.fit(X_train)
    X_train = imp.transform(X_train)

    # scale input so it is zero centered and smallish
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train) # what if I don't want to transform all variables?

    # transform the hold out dataset as well
    X_val_sc = scaler.transform(imp.transform(X_val)) 
    
    return X_scaled, X_val_sc, y_train, y_val, imp, scaler

def calculate_stats(y_pred, y_test):
    cm = metrics.confusion_matrix(y_test, y_pred)
    tn = np.float(cm[0][0]); fn = np.float(cm[1][0])
    tp = np.float(cm[1][1]); fp = np.float(cm[0][1])
    sens = tp / (tp + fn) if (tp + fn) != 0 else 0
    spec = tn / (tn + fp) if (tn + fp) != 0 else 0
    PPV = tp / (tp + fp) if (tp + fp) != 0 else 0
    NPV = tn / (tn + fn) if (tn + fn) != 0 else 0
    percent_pos = (tp + fp)/(tp + fp + fn + tn)
    return sens, spec, PPV, NPV, percent_pos

def search_thresh(y_prob, y_test):
    for thresh in np.arange(0.0, 1.0, 0.001):
        predictions = y_prob > thresh
        sens, spec, PPV, NPV, percent_pos = calculate_stats(predictions, y_test)
        if sens < 0.96:
            return thresh, sens, spec, PPV, NPV, percent_pos

def eval_model(model, X_test, y_test, thresh=None, plot=True):
    y_prob = model.predict_proba(X_test)[:, 1]
    if thresh == None:
        thresh, sens, spec, PPV, NPV, percent_pos = search_thresh(y_prob, y_test)
        predictions = y_prob > thresh
    else:
        predictions = y_prob > thresh
        sens, spec, PPV, NPV, percent_pos = calculate_stats(predictions, y_test)
    model_name = model.__class__.__name__
    print(model_name)
    print('AUPRC: {:.3f}'.format(metrics.average_precision_score(y_test, y_prob)))
    print('AUROC: {:.3f}'.format(metrics.roc_auc_score(y_test, y_prob)))
    print(metrics.confusion_matrix(y_test, predictions))
    print('sens: {:.3f} '.format(sens), 
            'spec: {:.3f} '.format(spec), 
            'PPV: {:.3f} '.format(PPV), 
            'NPV: {:.3f} '.format(NPV), 
            '%pos: {:.3f}'.format(percent_pos) )
    # print(metrics.classification_report(y_test, predictions))

    if plot:
        fig, ax = plt.subplots(1, 3, figsize=(12, 3))
        # AUROC, AUPRC
        metrics.plot_roc_curve(model, X_test, y_test, ax = ax[0]) 
        metrics.plot_precision_recall_curve(model, X_test, y_test, ax = ax[1])
        # calibration curve
        fraction_pos, mean_predicted_value = calibration_curve(y_test, y_prob, n_bins=20)
        ax[2].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        ax[2].plot(mean_predicted_value, fraction_pos, 's-', label=model_name)
        ax[2].set_xlabel('mean predicted value'); ax[2].set_ylabel('fraction positive')
        # save result
        plt.savefig('./result/' + model_name +'_AUC_plot.svg')
    return thresh

def fit_model(model, X_train, y_train, perform_cv = False):
    if perform_cv:
        param_test = {'min_samples_split':[2, 30, 100], 
                'max_depth':[None, 6, 9, 12], 
                'n_estimators':[100, 200]
                }
        grid = GridSearchCV(estimator=model, param_grid=param_test, scoring='average_precision', n_jobs=-1)
        grid_result = grid.fit(X_train, y_train)
        print(grid_result.best_params_)
        print(grid_result.best_score_)
        print(pd.DataFrame(grid_result.cv_results_[['params', 'mean_test_score', 'rank_test_score']]))
        return grid_result.best_estimator_
    else:
        model.fit(X_train, y_train)
        return model

def attach_historical_transfusion_data(df, df_historical):
    ''' Joins historical transfusion data with df
        - df_historical = training data to use for assigning percent_transfused
            i.e. to simulate forward prediction, should use transfusion data from year(s) prior
                 since current year data is not available at time of prediction
        - returns dataframe
    '''
    # reassign percent_transfused based on historical practice
    df_historical = df_historical.groupby('CPT').percent_transfused.mean().reset_index()
    df_historical['CPT'] = df_historical['CPT'].astype(str)
    df['CPT'] = df['CPT'].astype(str)
    df.drop('percent_transfused', axis=1, inplace=True)
    df = df.merge(df_historical, on='CPT', how='left')

    return df

def fit_NSQIP(random_state = 1, perform_cv=False):
    ''' Load NSQIP 2016-2018 > split 80/20 to training and testing > train 4 models on training set.
        Prints evaluation on NSQIP 2016-2018 test set (not reported in manuscript)
        Saves trained models as joblib pickle files
        Params:
            - random_state : set random state for model training
            - perform_cv : perform 5 fold crossvalidation using training split for hyperparameter tuning
                           need to have only 1 model in model list, and uncomment relevant hyperparameters
    '''
    # Read data
    data = pd.read_csv('puf16-18_lite_v4.csv')

    feat_to_exclude = ['PUFYEAR', 'CaseID', 'OTHBLEED', 'NOTHBLEED', 'DOTHBLEED', 'CPT', 'PRNCPTX',\
                       'count', 'HtoODay', 'INOUT', 'SDSA', 'EMERGNCY', 'PRWBC',
                       'ASA', # removed ASA score as a predictor
                       'BLEEDDIS', 'RENAFAIL', 'STEROID',
                       'NOTHBLEED_d0'
                       ]
    feat_used = list(set(data.columns.to_list()) - set(feat_to_exclude))
    print(feat_used)
    # feat_used = ['percent_transfused']

    # Fill NA, normalize, split to train, test, split
    X_train, X_val, y_train, y_val, imp, scaler = preprocess(data[feat_used], data.NOTHBLEED_d3)

    # Run in batch over a list of models
    model_list =  [ LogisticRegression(n_jobs=-1, solver='saga',
                                penalty = 'elasticnet', C = 0.01, l1_ratio = 1.0
                                ),
                    DecisionTreeClassifier(class_weight='balanced', max_depth=9, random_state = random_state),
                    RandomForestClassifier(n_jobs=-1, class_weight='balanced', n_estimators=200, random_state = random_state,
                                max_features = 5
                                ),
                    XGBClassifier(objective='binary:logistic', booster='gbtree', n_jobs=-1,
                                  random_state = random_state,
                                  learning_rate=0.05, n_estimators = 609, 
                                  colsample_bytree = 0.7, min_child_weight=4, max_depth = 6,
                                  )
                    ]

    for model in model_list:
        # Hyperparameter tuning
        if perform_cv:
            param_test = { # Logistic Regression
                'C':[0.03, 0.01, 0.003, 0.001],
                'l1_ratio': [0] #[0, 0.5, 1]
                }
            # param_test = { # Decision Tree
            #     'max_depth' : [8, 9, 10]
            #     }
            # param_test = { # Random Forest
            #     # 'max_depth':[None, 5, 10], 
            #     # 'min_samples_split':[2, 20, 200], 
            #     'max_features':[4, 5, 6] 
            #     }
            # param_test = { # XGBoost
            #     'min_child_weight':[1, 3, 4, 5], 
            #     'max_depth': [4, 5, 6, 7], 
            #     'gamma':[0, 0.05, 0.1], 
            #     'subsample':[1, 0.9, 0.8],
            #     'colsample_bytree':[0.75, 0.7, 0.6],
            #     }

            # perform grid search over available options
            grid = GridSearchCV(estimator=model, param_grid=param_test, scoring='average_precision', n_jobs=-1)
            grid_result = grid.fit(X_train, y_train)
            print(grid_result.best_params_)
            print(grid_result.best_score_)
            print(pd.DataFrame(grid_result.cv_results_)[['params', 'mean_test_score', 'rank_test_score']])
            
            # set parameters to optimal parameters
            params = model.get_params()
            for k in grid_result.best_params_:
                params[k] = grid_result.best_params_[k]
            model.set_params(**params)

        # Fit on NSQIP 2016-2018 training set
        print('Test performance')
        model = fit_model(model, X_train, y_train)
        
        # Evaluate on NSQIP 2016-2018 test set, finding threshold for 95% sensitivity
        thresh = eval_model(model, X_val, y_val)
        print('threshold: {:.4f}'.format(thresh))

        # Save model
        data_pipeline = ProcessData(scaler, imp, None, None)
        data_pipeline.save_params(False, True, True, feat_used, thresh)
        data_pipeline.save_model(model)
        model_name = model.__class__.__name__
        joblib.dump(data_pipeline, './result/' + model_name + '_pipeline.joblib')

        # Print parameters
        print('--------------------------')
        print(model.get_params())
        print('--------------------------\n\n')

if __name__ == '__main__':
    fit_NSQIP()