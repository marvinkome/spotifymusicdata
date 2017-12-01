#Author: Marvin kome
import pickle
from feature_format import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 

def featureExt(pklfile, featureList,  remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False):
    #Extract Pickle data
    data = pickle.load(open(pklfile ,'r'))

    #print data['d0']
    
    #Extract attributes
    dataArray = featureFormat(data, featureList,  remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False)
    
    #Seperate label and features
    label, features = targetFeatureSplit(dataArray)
    
    #Scale The Data
    scaler = StandardScaler()
    features = scaler.fit_transform(features) 
    
    #Seperating Training and testing data
    diff_amount = int(round(len(features) * 0.75))
    features_train = features[:diff_amount]
    labels_train = label[:diff_amount]
    features_test = features[diff_amount:]
    labels_test = label[diff_amount:]
        
    return features_train, labels_train, features_test, labels_test, dataArray

feature_list = ['target','energy','mode']
featureExt('musicData.pkl', feature_list)
