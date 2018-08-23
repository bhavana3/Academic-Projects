# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 22:22:44 2016

@author: Bhavana
"""
import numpy as np
import pandas as pd

from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC

#==============================================================================
# Input - classifier, dataset
# Output - nothing
# Description - This method applies the startified k fold after removing outliers only from training data
# and then print each fold scores and finally mean score
# referenced syntax from "http://stackoverflow.com/"
#==============================================================================
def Calculate_Stratifiedaccuracy(clf, nba_feature, nba_class):
    scores = []
    # cross-validation
    kfold = StratifiedKFold(n_splits=10)
    for train_index, test_index in kfold.split(nba_feature, nba_class):
        # split the training and testing data
        X_train, X_test = nba_feature.ix[train_index], nba_feature.ix[test_index]
        y_test = nba_class[test_index]
        indexes = X_train.index[X_train["MP"] > 4]
        X_train1 = nba_feature.ix[indexes]
        y_train1 = nba_class[indexes]
        clf.fit(X_train1, y_train1)
        scores.append(clf.score(X_test, y_test))
    
    print("After removing outliers from training data, scores: {}" .format(scores))
    # calculating mean score for folds
    print("Avergae Accuracy: %0.2f" % np.mean(scores))

#read from the csv file and return a Pandas DataFrame. - code referred from given document
nba_stats = pd.read_csv('NBAstats.csv')
# "Position (pos)" is the class attribute we are predicting - code referred from given document
class_column = 'Pos'

#The dataset contains many attributes 
#We know that sokme of them are not useful for classification and thus do not 
#include them as features. 
#===========code referred from given document=============
feature_columns = ['Age', 'MP', 'FG', 'FGA', '3P', '3PA', \
    '2P', '2PA', '2P%', 'eFG%', 'FT%', 'ORB', 'DRB', \
    'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PS/G']
    
#Pandas DataFrame allows you to select columns. 
#We use column selection to split the data into features and class.
#===========code referred from given document=============
nba_feature = nba_stats[feature_columns]
nba_class = nba_stats[class_column]

print(nba_class[0])

#===========Split data into testing and training using method available in package 
#(code referred from given document)=============
train_feature, test_feature, train_class, test_class = \
    train_test_split(nba_feature, nba_class, stratify=nba_class, \
    train_size=0.75, test_size=0.25)

#===========syntax referred from "http://scikit-learn.org/" =================
linearsvm = OneVsOneClassifier(estimator=LinearSVC(dual = False,class_weight="balanced"))
linearsvm.fit(train_feature, train_class)
scores = cross_val_score(linearsvm, nba_feature, nba_class, cv=10, scoring='accuracy')
#print("linearsvm - Test set score: {:.3f}".format(linearsvm.score(test_feature, test_class)))
#print("Cross-validation scores: {}".format(scores))
#print("Average Accuracy: %0.2f" % (scores.mean()))
#Calculate_Stratifiedaccuracy(linearsvm, nba_feature, nba_class)

#===========code referred from given document=============
train_feature, test_feature, train_class, test_class = \
    train_test_split(nba_feature, nba_class, stratify=nba_class, \
    train_size=0.75, test_size=0.25)

#===========syntax referred from "http://scikit-learn.org/" =================
clf = RandomForestClassifier(n_estimators=100, max_depth=10, criterion='entropy')
clf.fit(train_feature, train_class)
scores = cross_val_score(clf, nba_feature, nba_class, cv=10, scoring='accuracy')
#print("####################")
#print()
#print("Random Forest Classifier - Test set score: {:.3f}".format(clf.score(test_feature, test_class)))
#print("Cross-validation scores: {}".format(scores))
#print("Average Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
#Calculate_Stratifiedaccuracy(clf, nba_feature, nba_class)

#===========code referred from given document=============
feature_columns = ['Age', 'MP', 'FG', 'FGA', '3P', '3PA', \
    '2P', '2PA', 'eFG%', 'FT%', 'ORB', 'DRB', \
    'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PS/G']
nba_feature = nba_stats[feature_columns]

#===========code referred from given document=============
train_feature, test_feature, train_class, test_class = \
    train_test_split(nba_feature, nba_class, stratify=nba_class, \
    train_size=0.75, test_size=0.25)
    
#===========syntax referred from "http://scikit-learn.org/" =================
clf1 = LogisticRegression(class_weight ="balanced", solver='newton-cg', multi_class='multinomial')
clf1.fit(train_feature, train_class)
scores = cross_val_score(clf1, nba_feature, nba_class, cv=10, scoring='accuracy')
#print("####################")
#print()
#print("LogisticRegression Classifier - Test set score: {:.3f}".format(clf1.score(test_feature, test_class)))
#print("Cross-validation scores: {}".format(scores))
#print("Average Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
#Calculate_Stratifiedaccuracy(clf1, nba_feature, nba_class)

#===========code referred from given document=============
train_feature, test_feature, train_class, test_class = \
    train_test_split(nba_feature, nba_class, stratify=nba_class, \
    train_size=0.75, test_size=0.25)

    
#===========syntax referred from "http://scikit-learn.org/" =================
eclf = VotingClassifier(estimators=[('linearsvm', linearsvm), ('RFC', clf), ('lr', clf1)], voting='hard')
eclf.fit(train_feature, train_class)
print("####################")
print()
print("VotingClassifier - Test set score: {:.3f}".format(eclf.score(test_feature, test_class)))
prediction = eclf.predict(test_feature)
#===========code referred from given document=============
print("Confusion matrix:")
print(pd.crosstab(test_class, prediction, rownames=['True'], colnames=['Predicted'], margins=True))
scores = cross_val_score(eclf, nba_feature, nba_class, cv=10, scoring='accuracy')
print("Cross-validation scores: {}".format(scores))
print("Accuracy: Ensemble %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
Calculate_Stratifiedaccuracy(eclf, nba_feature, nba_class)

#===========save the training set into a CSV file 
#code referred from given document=============
train_class_df = pd.DataFrame(train_class,columns=[class_column])     
train_data_df = pd.merge(train_class_df, train_feature, left_index=True, right_index=True)
train_data_df.to_csv('train_data.csv', index=False)
#===========save the testing set into a CSV file 
#code referred from given document=============
temp_df = pd.DataFrame(test_class,columns=[class_column])
temp_df['Predicted Pos']=pd.Series(prediction, index=temp_df.index)
test_data_df = pd.merge(temp_df, test_feature, left_index=True, right_index=True)
test_data_df.to_csv('test_data.csv', index=False)

prediction = classifier.prob_classify()
    #===========code referred from given document=============
    print("Confusion matrix:")
    print(pd.crosstab(test_set, prediction, rownames=['True'], colnames=['Predicted'], margins=True))