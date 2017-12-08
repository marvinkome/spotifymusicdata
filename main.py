from preprocess import *
from tools import *
import pickle

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor


from outlier_cleaner import outlierCleaner

import numpy

def main():
    
    '''Runs the main training and testing'''
    
    regressor = RandomForestRegressor(n_estimators=70)

    ### Features List
    ### Features List
    featureList = ['acousticness', 'energy', 'liveness', 'speechiness', 'key',
                       'instrumentalness', 'mode', 'time_signature', 'duration_ms', 'loudness',
                       'valence', 'danceability','artist']
    
    label = 'speechiness'

    featureList2 = ['liveness']


    ### Extracting Training and Testing data
    features_train, labels_train, features_test, labels_test = data_preprocess('data.csv',featureList,label)
    
    
    ### Training the data and returning the classifier name and the accuracy
    acc, pred, reg = trainData(regressor, features_train,labels_train,features_test, labels_test)

    ### Printing Accuracy
    print 'No. Training Data:  {0}'.format(features_test.shape)
    print 'No. Testing Data: {0}'.format(labels_test.shape)
    print 'Orig Data: {0}'.format(labels_test[0:5])
    print 'Predictions: {0}'.format(pred[0:5])
    print 'score: {0}'.format(acc)

    pickle.dump(reg,open('bestreg.pkl', "w"))

    '''for i in range(13):
        for feature, target in zip(features_test[i], labels_test):
            plt.scatter( feature, target, color='c' ) '''
            
    #plt.plot( features_test, labels_test, color='r')
    #plt.plot( features_test, pred, color='g')
    #plt.show()


if __name__ == '__main__':
    main()


    
