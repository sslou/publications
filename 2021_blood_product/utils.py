import numpy as np
import pandas as pd
from sklearn import metrics
from map_cpt_to_text import map_cpt_to_name, map_cpt_to_ancestor

class ProcessData:
    def __init__(self, normalizer, imputer, vectorizer, pca):
        self.normalizer = normalizer
        self.imputer = imputer
        self.vectorizer = vectorizer
        self.pca = pca
        self.model = None
        self.feat_importance = None
        self.best_threshold = None
    def save_model(self, model):
        self.model = model
    def save_importance(self, importance):
        self.feat_importance = importance
    def save_params(self, use_words, use_static, perform_impute, feat_used, best_threshold):
        self.use_words = use_words
        self.use_static = use_static
        self.perform_impute = perform_impute
        self.feat_used = feat_used
        self.best_threshold = best_threshold

def load_BJH_data(data_filename, prefilter=None):
    ''' Load BJH data from data_filename
        - prefilter (int) - can be used to exclude cases occuring fewer than `prefilter` times
            this is relevant if the percent_transfused data is noisy due to a rarely performed surgery
            common values: 50, 30
        - returns dataframe 
    '''
    data = pd.read_csv(data_filename, encoding='cp1252')
    data.rename(columns={'AN_PROC_NAME':'CPT_name'}, inplace=True)
    if prefilter:
        data = data.loc[data['count'] > prefilter, :]
    return data

def load_NSQIP_data(data_filename, historical_filename='puf16-18_lite_v3.csv', forward=True):
    ''' Loads NSQIP PUF data from data_filename
        - data_filename = name of puf*_lite.csv for which predictions are desired
        - historical_filename = name of puf*_lite.csv to retrieve percent_transfused from
            i.e. to simulate forward prediction, should use transfusion data from year(s) prior
                 since current year data is not available at time of prediction
        - forward = set to False if you want to skip historical percent_transfused assignment
        - returns dataframe
    '''
    df = pd.read_csv(data_filename)

    # map cpt to words
    cpt = pd.read_table('CPT4_procedures.tsv')
    puf_wnames = map_cpt_to_name(df, cpt)
    cpt_ancestor = pd.read_table('CPT4_ancestors_leaf.tsv')
    df = map_cpt_to_ancestor(puf_wnames, cpt_ancestor) # adds CPT_name, CPT_group features

    # reassign percent_transfused based on historical practice
    if forward:
        df_historical = pd.read_csv(historical_filename)
        df_historical = df_historical.groupby('CPT').percent_transfused.mean().reset_index()
        df_historical['CPT'] = df_historical['CPT'].astype(str)
        df['CPT'] = df['CPT'].astype(str)
        df.drop('percent_transfused', axis=1, inplace=True)
        df = df.merge(df_historical, on='CPT', how='left')

    return df

def metric_eval(y_test, y_pred):
    C = metrics.confusion_matrix(y_test, y_pred)
    tn = np.float(C[0][0])
    fn = np.float(C[1][0])
    tp = np.float(C[1][1])
    fp = np.float(C[0][1])

    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    PPV = tp / (tp + fp) if (tp + fp) != 0 else 0
    NPV = tn / (tn + fn) if (tn + fn) != 0 else 0
    acc = metrics.accuracy_score(y_test, y_pred)
    frac_pos = (tp + fp) / (tp + fn + fp + tn)

    return sensitivity, specificity, PPV, NPV, frac_pos

def line_search_best_metric(y_test, y_prob, sens_thresh=0.96):
    for t in np.arange(0.0, 1.0, 0.001):
        sens, spec, PPV, NPV, acc = metric_eval(y_test, y_prob > t)
        if sens < sens_thresh:
            return (sens, spec, PPV, NPV, acc), t

def calculate_metrics(y_test, y_prob, best_threshold = None, verbose = False):
    fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)
    prec, rec, _ = metrics.precision_recall_curve(y_test, y_prob)
    auroc = metrics.auc(fpr, tpr)
    auprc = metrics.auc(rec, prec)
    brier = metrics.brier_score_loss(y_test, y_prob)
    if best_threshold == None:
        (sens, spec, PPV, NPV, frac_pos), best_threshold = line_search_best_metric(y_test, y_prob)
    else:
        (sens, spec, PPV, NPV, frac_pos) = metric_eval(y_test, y_prob > best_threshold)
    if verbose:
        brier = metrics.brier_score_loss(y_test, y_prob)
        print('brier:', brier)
        print('threshold:', best_threshold)
    return (auroc, auprc, sens, spec, PPV, frac_pos, brier)