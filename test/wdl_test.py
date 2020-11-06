import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from ctr_model.models import WDL
from ctr_model.inputs import SparseFeat, DenseFeat, get_feature_names

import torch

if __name__ == "__main__":
    data = pd.read_csv('../dataset/criteo_sample.txt')

    sparse_features = ['C'+ str(i) for i in range(1, 27)]
    dense_features = ['I'+ str(i) for i in range(1, 14)]

    # 补充缺失值
    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # sparse -> label dense -> 0~1
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # count #unique features for each sparse field,and record dense feature field name
    feature_columns = [SparseFeat(feat, data[feat].unique()) for feat in sparse_features] + \
                             [DenseFeat(feat,1,) for feat in dense_features]

    dnn_feature_columns = feature_columns
    linear_feature_columns = feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # generate train and eval data for model
    train, test = train_test_split(data, test_size=0.2)
    train_model_input = [train[name] for name in feature_names]
    test_model_input = [test[name] for name in feature_names]

    # define model, tarin, eval, predict
    device = "cpu"
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print("cuda ready")
        device = 'cuda:0'

    model = WDL(linear_feature_columns, dnn_feature_columns,task='binary',
                l2_reg_embedding=1e-5,l2_reg_linear=1e-5,l2_reg_dnn=0,device=device)

    model.compile("adagrad", "binary_crossentropy", metrics=["binary_crossentropy", "auc"], )
    model.fit(train_model_input, data[target].values, batch_size=256, epochs=10, validation_split=0.2, verbose=2)

    pred_ans = model.predict(test_model_input, 256)

    print("test logloss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))