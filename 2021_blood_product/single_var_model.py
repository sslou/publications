import numpy as np
from utils import *
import joblib

class SK_model:
    def __init__(self, var_index):
        '''var_index is the column index for the single variable to be used'''
        self.var_index = var_index
    def predict_proba(self, data):
        data = np.array(data)
        y_pred = data[:, self.var_index]
        # rescale btwn zero and one (to ~ undo normalization and scaling to unit variance)
        y_pred = (y_pred - min(y_pred))/(max(y_pred) - min(y_pred))
        pred_0 = 1 - y_pred
        return np.stack((pred_0, y_pred), axis=1)
    def get_params(self):
        return {'var_index':self.var_index}


if __name__ == '__main__':
    data_train = load_NSQIP_data('puf16-18_lite_v3.csv', forward = False)
    y_train = data_train.NOTHBLEED_d3
    
    # borrow preprocessing pipeline from LR
    LR_pipe = joblib.load('./model/LogisticRegression_pipeline.joblib')
    feat_used = LR_pipe.feat_used
    imputer = LR_pipe.imputer
    scaler = LR_pipe.normalizer

    # preprocess
    data_train = data_train[feat_used]
    data_train = imputer.transform(data_train)
    data_train = scaler.transform(data_train)

    # specify model
    model = SK_model(feat_used.index('percent_transfused'))

    # evaluate
    y_prob = model.predict_proba(data_train)[:, 1]
    result = calculate_metrics(y_train, y_prob, best_threshold=None, verbose=True)
    print(result)

    # save
    pipeline = ProcessData(scaler, imputer, None, None)
    pipeline.save_params(False, True, True, feat_used, 0.011)
    pipeline.save_model(model)
    joblib.dump(pipeline, './result/baseline_pipeline.joblib')