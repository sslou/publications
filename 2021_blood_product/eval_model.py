from utils import *
from single_var_model import SK_model
import sklearn
import random

def evaluate(model, X_test, y_test, best_threshold=None, plot=False):
    # Testing
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluation
    fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)
    prec, rec, _ = metrics.precision_recall_curve(y_test, y_prob)
    if best_threshold == None:
        (sensitivity, specificity, PPV, NPV, frac_pos), best_threshold = line_search_best_metric(y_test, y_prob)
    else:
        (sensitivity, specificity, PPV, NPV, frac_pos) = metric_eval(y_test, y_prob > best_threshold)
    y_pred = y_prob > best_threshold

    print('--------------------------------------------')
    print('Evaluation of test set:')
    print("AU-ROC:", "%0.4f" % metrics.auc(fpr, tpr),
          "AU-PRC:", "%0.4f" % metrics.auc(rec, prec))
    print("sensitivity:", "%0.4f" % sensitivity,
          "specificity:", "%0.4f" % specificity,
          "PPV:", "%0.4f" % PPV,
          "NPV:", "%0.4f" % NPV,
          "fraction pos:", "%0.4f" % frac_pos)
    print(metrics.confusion_matrix(y_test, y_pred))
    print('--------------------------------------------')

    # plot and save figs
    if plot:
        fig, ax = plt.subplots(1, 3, figsize = (15, 4))
        ax[0].plot(fpr, tpr); ax[0].set_title('AUROC')
        ax[1].plot(rec, prec); ax[1].set_title('AUPRC')
        fraction_pos, mean_predicted_value = calibration_curve(y_test, y_prob, n_bins=20)
        ax[2].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        ax[2].plot(mean_predicted_value, fraction_pos, 's-')
        ax[2].set_xlabel('mean predicted value'); ax[2].set_ylabel('fraction positive')
        plt.savefig('result/AUC_plot.svg')
    return best_threshold

def run_model(data_pipeline, data_source, prefilter = None):
    ''' using model pipeline from joblib, loads dataset specified by data_source, runs model,
        and evaluates performance using original threshold from training as well as recomputing
        a threshold to attain sensitivity 0.96
    '''
    # define parameters
    use_static = data_pipeline.use_static
    perform_impute = data_pipeline.perform_impute
    use_words = data_pipeline.use_words
    best_threshold = data_pipeline.best_threshold
    feat_used = data_pipeline.feat_used
    print('use_static:', use_static)
    print('use_words:', use_words)
    print(feat_used)
    print('prefilter', prefilter)

    # assign predictors and outcomes
    if 'BJH' in data_source:
        test_data = load_BJH_data(data_source, prefilter)
        y_test = test_data.NOTHBLEED
    else:
        test_data = load_NSQIP_data(data_source, forward = True)
        y_test = test_data.NOTHBLEED_d3
        # y_test = test_data.NOTHBLEED
    test_static = test_data[feat_used]
    test_text = test_data['CPT_name']
    
    # transform data
    if perform_impute:
        test_static, _ = perform_imputation(test_static, imputer=data_pipeline.imputer)
    else:
        data_imputed = test_static
    X_test = data_pipeline.normalizer.transform(data_imputed)

    # predict and evaluate
    print('eval w original model threshold')
    evaluate(data_pipeline.model, X_test, y_test, best_threshold)
    print('eval w re-tuned threshold for sens 0.96')
    new_thresh = evaluate(data_pipeline.model, X_test, y_test, best_threshold = None)
    return data_pipeline.model, X_test, y_test, new_thresh

def bootstrap_stats(model, X_test, y_test, best_threshold = None, data_size =100000):
    y_test = np.array(y_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    observed_result = pd.Series(calculate_metrics(y_test, y_prob, best_threshold), 
        index = ['AUROC', 'AUPRC', 'Sensitivity', 'Specificity', 'PPV', 'FractionPos', 'Brier'], name='Observed')
    boot_results = []
    # run bootstrapping
    for i in range(1000):
        rand_i = [random.randint(0, len(y_prob)-1) for i in range(data_size)]
        y_prob_r = y_prob[rand_i]
        y_test_r = y_test[rand_i]
        boot_results.append(calculate_metrics(y_test_r, y_prob_r, best_threshold))
    
    # summarize results
    boot_results = pd.DataFrame(boot_results, 
        columns = ['AUROC', 'AUPRC', 'Sensitivity', 'Specificity', 'PPV', 'FractionPos', 'Brier'])
    return pd.concat([observed_result, boot_results.quantile(0.025), boot_results.quantile(0.975)], axis=1)

def plot_auc(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)
    prec, rec, _ = metrics.precision_recall_curve(y_test, y_prob)
    obs_pos, pred_pos = calibration_curve(y_test, y_prob, n_bins=20)
    return fpr, tpr, prec, rec, obs_pos, pred_pos

def main(data_pipeline, model_name = '', bootstrap = False, plot=False):
    ''' Input: joblib saved model pipeline
        Can perform bootstrap resampling of validation datasets
        Can also plot overlapped calibration and AUC curves
    '''
    data_source = 'puf19_lite_v3.csv'
    model, X_test, y_test, new_threshold = run_model(data_pipeline, data_source, prefilter = 0)
    print('PUF19_threshold', data_pipeline.best_threshold)
    if bootstrap:
        bootstrap_cis = bootstrap_stats(model, X_test, y_test, data_pipeline.best_threshold).add_prefix('PUF19_')
        # print(bootstrap_cis)
    if plot:
        fpr, tpr, prec, rec, obs_pos, pred_pos = plot_auc(model, X_test, y_test)

    data_source = './raw_data/2020_BJH_Labs_Notes.csv'
    model, X_test, y_test, new_threshold = run_model(data_pipeline, data_source, prefilter = 50)
    print('BJH threshold', new_threshold)

    if bootstrap:
        bootstrap_cis2 = bootstrap_stats(model, X_test, y_test, 
            best_threshold=new_threshold, data_size = 20000).add_prefix('BJH50_')
        output = pd.concat([bootstrap_cis, bootstrap_cis2], axis=1)
        output.to_csv('./result/result_bootstrap_' + model_name + '.csv')
    if plot:
        fpr1, tpr1, prec1, rec1, obs_pos1, pred_pos1 = plot_auc(model, X_test, y_test)
        fig, ax = plt.subplots(1, 3, figsize=(12, 3))
        ax[0].set_title('AUROC delta')
        ax[1].set_title('AUPRC delta')
        ax[2].set_title('Calibration')
        ax[2].set_xlabel('Predicted probability'); ax[2].set_ylabel('Observed probability')
        ax[2].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        ax[0].plot(fpr, tpr) #AUC: (fpr, tpr)
        ax[0].plot(fpr1, tpr1)
        ax[1].plot(rec, prec)
        ax[1].plot(rec1, prec1)
        ax[2].plot(pred_pos, obs_pos, 's-')
        ax[2].plot(pred_pos1, obs_pos1, 's-')
        fig.savefig('./result/auc_cal_' + model_name + '.svg')

        return (fpr, tpr, rec, prec, fpr1, tpr1, rec1, prec1)

if __name__ == '__main__':
    model_list = {  
                    'baseline' : './model/baseline_pipeline.joblib',
                    'LR' : './model/LogisticRegression_pipeline.joblib',
                    'DT' : './model/DecisionTreeClassifier_pipeline.joblib',
                    # 'RF' : './model/RandomForestClassifier_pipeline.joblib',
                    'XGB': './model/XGB_pipeline.joblib',
                    }
    curves = []
    for name in model_list:
        data_pipeline = joblib.load(model_list[name])
        c = main(data_pipeline, name, bootstrap = True, plot=True)
        curves.append(c)

