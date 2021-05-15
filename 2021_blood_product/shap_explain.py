from utils import *
import shap
import joblib
import matplotlib.pyplot as plt
import math

def predict_example(model, imputer, scaler, vec):
    ''' Given a model and individual slice of input data, i.e. df[i, :],
        compute the predicted probability of transfusion
    '''
    vec = np.array(vec).reshape(1, -1)
    if imputer != None:
        vec = imputer.transform(vec)
    vec = scaler.transform(vec)
    y_prob = model.predict_proba(vec)[:, 1]
    return y_prob[0]

def show_input_data(i):
    vec = data.iloc[i, :]
    orig_rownum = vec.name
    y_prob = predict_example(model, imputer, scaler, data[feat_used].iloc[i, :])
    print('CPT: ', vec.CPT_name)
    print(vec)
    print('prediction: ', y_prob)
    return vec, y_prob

def single_force_plot(i, html=True):
    if html:
        fig = shap.force_plot(explainer.expected_value, shap_values[i, :], data_to_explain.iloc[i, :],
            feature_names=feat_used, show=False, link='logit')
        shap.save_html('./result/shap_force_plot_' + str(i) + '.htm', fig)
    else:
        fig = shap.force_plot(explainer.expected_value, shap_values[i, :], data_to_explain.iloc[i, :],
            feature_names=feat_used, show=False, matplotlib=True, link='logit')
        # fig = plt.gcf()
        # fig.savefig('./result/shap_force_plot_' + str(i) + '.svg')
        # fig.close()
    return fig


if __name__ == '__main__':
    data_pipeline = joblib.load('model/XGB_pipeline.joblib')
    # data_to_explain = load_NSQIP_data('puf16-18_lite_v3.csv', forward = False)
    data = load_BJH_data('./raw_data/2020_BJH_Labs_Notes.csv', prefilter=50)

    use_static = data_pipeline.use_static
    perform_impute = data_pipeline.perform_impute
    use_words = data_pipeline.use_words
    best_threshold = data_pipeline.best_threshold
    feat_used = data_pipeline.feat_used
    model = data_pipeline.model
    imputer = data_pipeline.imputer
    scaler = data_pipeline.normalizer

    data_to_explain = data[feat_used]
    data_to_explain = pd.DataFrame(imputer.transform(data_to_explain), columns=feat_used)
    data_scaled = pd.DataFrame(scaler.transform(data_to_explain), columns=feat_used)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data_scaled)

    # force plot for individual example
    fig = single_force_plot(100, html=True)
    # wrong examples: 3117, 601, 437

    # shap summary plot
    # plt.figure()
    # shap.summary_plot(shap_values, data[feat_used])
    # plt.savefig('./result/shap_summary.svg')

    # shap dependence plots
    # shap.dependence_plot("PRHCT", shap_values, data_to_explain, interaction_index="percent_transfused")
    # plt.savefig('./result/shap_HCT.svg')
    # shap.dependence_plot("percent_transfused", shap_values, data_to_explain, interaction_index="PRHCT")
    # plt.savefig('./result/shap_percent_transfused.svg')