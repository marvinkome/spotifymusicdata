import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Imputer, LabelEncoder
from sklearn.feature_selection import VarianceThreshold

import numpy as np

def data_preprocess(url,feat_cols, lab_cols):

    tt = pd.read_csv(url)

    for col in feat_cols:
        if type(col) == 'float' or type(col) == 'int':
            imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
            tt[col] = imp.fit_transform(tt[col])
        else:
            le = LabelEncoder()
            tt[col] = le.fit_transform(tt[col])



    feat_scaler = MinMaxScaler()
    tt[feat_cols] = feat_scaler.fit_transform(tt[feat_cols])

    features = tt[feat_cols].values
    labels = tt[lab_cols].values

    #Seperating Training and testing data
    diff_amount = int(round(len(features) * 0.75))
    features_train = features[:diff_amount]
    labels_train = labels[:diff_amount]
    features_test = features[diff_amount:]
    labels_test = labels[diff_amount:]

    return features_train, labels_train, features_test, labels_test

        
