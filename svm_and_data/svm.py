#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 20:58:55 2021

@author: kev
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from imblearn.over_sampling import RandomOverSampler 

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

# read data
xTrain = pd.read_csv('xTrain.csv', index_col=0)
xTest = pd.read_csv('xTest.csv', index_col=0)

yTrain_cat = pd.read_csv('CateyTrain.csv', index_col=0)
yTest_cat = pd.read_csv('CateyTest.csv', index_col=0)
yTrain_1 = yTrain_cat['1']
yTrain_5 = yTrain_cat['0.5']
yTrain_2 = yTrain_cat['0.2']
yTest_1 = yTest_cat['1']
yTest_5 = yTest_cat['0.5']
yTest_2 = yTest_cat['0.2']


# look at the distribution of the target feature
distribution = np.unique(yTrain_1, return_counts = True)
print(distribution)
perc_45 = distribution[1][3]/np.sum(distribution[1])
print(perc_45)


# oversampling
ros = RandomOverSampler(random_state = 334)
xTrain_res, yTrain_res = ros.fit_resample(xTrain, yTrain_1)

# SVM
# tune C and gamma for the svc model
grid = {
    'C':            [0.1, 1, 10, 100, 1000],
    'gamma':        [1, 0.1, 0.01, 0.001, 0.0001],
    }

svc = SVC()
grid = GridSearchCV(estimator = svc, param_grid = grid, 
                          cv = 5, refit = True, verbose = 2)
grid.fit(xTrain_res, yTrain_res)
print(grid.best_params_)

# Run the model with optimal params resampled
svm = SVC(C = 1000, gamma=1, kernel = 'rbf')
svm.fit(xTrain_res, yTrain_res)
yHat = svm.predict(xTest)
acc = accuracy_score(yHat, yTest_1)
f1 = f1_score(yTest_1, yHat, average='weighted')
print("svm res", acc)
print("svm res f1", f1)

# computing precision, recall and f1 score for each class
truth = np.zeros(4)
predict = np.zeros(4)
correct = np.zeros(4)
classes = {'1-2': 0, '2-3': 1, '3-4': 2, '4-5': 3}
for i in range(len(xTest)):
    y = yTest_1[i]
    predict[classes[yHat[i]]] += 1
    truth[classes[yTrain_1[i]]] += 1
    if (classes[yHat[i]] == classes[yTrain_1[i]]):
        correct[classes[yHat[i]]] += 1

print(correct)
print(predict)
print(truth)

precision = correct / predict
recall = correct / truth
f1 = 2 * (precision * recall) / (precision + recall)
print(precision)
print(recall)
print(f1)

