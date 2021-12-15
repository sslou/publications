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

    model = data_pipeline.model
    imputer = data_pipeline.imputer
    scaler = data_pipeline.normalizer
    feat_used = data_pipeline.feat_used

    # this is made up patient data
    example_data = {'PRSODM' : 140, 'PRALBUM' : 4.0, 'DIALYSIS' : 0, 'PRPLATE' : 100, 'HYPERMED' : 1,
              'HEIGHT' : 65, 'PRHCT' : 30, 'PRCREAT' : 1.0 , 'PRPTT' : 33, 'HXCHF' : 0, 'WEIGHT' : 190,
              'ELECTSURG' : 1, 'HXCOPD' : 1, 'PRBILI' : 1.0, 'SEX' : 1, 'percent_transfused': 1.3,
              'PRINR' : 1, 'DIABETES' : 1, 'SMOKE' : 1, 'Age': 70}
    example_data = pd.DataFrame(example_data, index=[0])

    # transform data
    data_to_explain = example_data[feat_used]
    data_to_explain = pd.DataFrame(imputer.transform(data_to_explain), columns=feat_used)
    data_scaled = pd.DataFrame(scaler.transform(data_to_explain), columns=feat_used)
    print('model predicted probability: ', predict_example(model, imputer, scaler, example_data.loc[0, feat_used]))

    # calculate SHAP values and visualize
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data_scaled)
    single_force_plot(0)


