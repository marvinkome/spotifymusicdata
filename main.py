from featureExtract import *
from outlier_cleaner import *
from tools import *

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def main():
    
    '''Runs the main training and testing'''
    names = ["gini",'entropy']
    
    classifiers = [RandomForestClassifier(
                                        max_features=0.9, n_estimators=350,
                                        criterion='entropy')

    ### Features List
    featureList = ['target','acousticness', 'energy', 'liveness', 'tempo', 'speechiness', 'key',
                       'instrumentalness', 'mode', 'time_signature', 'duration_ms', 'loudness',
                       'valence', 'danceability']

    featureList2 = ['target','instrumentalness','loudness']


    ### Extracting Training and Testing data
    features_train, labels_train, features_test, labels_test, data = featureExt('musicData.pkl', featureList)
    
    
    ### Training the data and returning the classifier name and the accuracy
    acc, bclf, clf = trainData(classifiers, names, features_train,labels_train,features_test,labels_test)

    ### Printing Accuracy
    print 'No. Training Data:  {feat}'.format(feat = len(features_train))
    print 'No. Testing Data: {feat}'.format(feat = len(features_test))
    print 'The Best is {clf} with {acc}% accuracy'.format(acc = acc, clf = bclf)

if __name__ == '__main__':
    main()
    
#Highest Accuracy so far is 75.0% with RandomForestClassifier(n_estimators=40)


    
