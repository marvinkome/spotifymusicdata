### Author Marvin
### The tools used in main.py defined here

import numpy as np

import matplotlib.pyplot as plt
import pylab as pl

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score

def trainData(classifiers, names, features_train, labels_train, features_test, labels_test):

    acc = []
    z = 0
    
    for clf in classifiers:
        #Naive Bayes
        #Fit and trains data
        clf.fit(features_train, labels_train)
        #Predict and returns accuracy
        pred = clf.predict(features_test)
        accu =  round(accuracy_score(pred, labels_test) * 100,2)
        acc.append(accu)

    for i in acc:
        if i == max(acc):
            bestClf = names[z]
            bestClfObj = classifiers[z]
        z += 1
           
    return max(acc), bestClf, bestClfObj

def importantFeatures(clf, features_train, featureList):
    ### Printing features importance
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    print("Feature ranking:")
    for f in range(features_train.shape[1]):
        c = indices[f]
        print("{f}. feature {d} ({i})" .format(f = f + 1, d = featureList[c+1], i = importances[indices[f]]))

    ### Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(features_train.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(features_train.shape[1]), indices)
    plt.xlim([-1, features_train.shape[1]])
    plt.show()

def decisionBoundary(clf, features_test, labels_test):
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    # step size in the mesh
    
    xx, yy = np.meshgrid(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.ylim(yy.min(), yy.max())
    plt.xlim(xx.min(), xx.max())
    
    plt.pcolormesh(xx, yy, Z, cmap=pl.cm.seismic)
    
    # Plot also the test points
    xx_t0 = [features_test[ii][0] for ii in range(0, len(features_test)) if labels_test[ii] == 0]
    xx_t1 = [features_test[ii][0] for ii in range(0, len(features_test)) if labels_test[ii] == 1]

    yy_t0 = [features_test[ii][1] for ii in range(0, len(features_test)) if labels_test[ii]==0]
    yy_t1 = [features_test[ii][1] for ii in range(0, len(features_test)) if labels_test[ii]==1]
    
    plt.scatter(xx_t0, yy_t0, color = "b", label="hate")
    plt.scatter(xx_t1, yy_t1, color = "r", label="like")

    plt.legend()
    plt.show()
        
