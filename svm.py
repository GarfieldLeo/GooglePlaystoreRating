#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 20:58:55 2021

@author: kev
"""

import pandas as pd
import numpy as np
import warnings
# warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

from imblearn.over_sampling import RandomOverSampler 
from imblearn.over_sampling import SMOTE 

xTrain = pd.read_csv('xTrain.csv', index_col=0)
yTrain = pd.read_csv('yTrain.csv', index_col=0)
xTest = pd.read_csv('xTest.csv', index_col=0)
yTest = pd.read_csv('yTest.csv', index_col=0)

yTrain_cat = pd.read_csv('CateyTrain.csv', index_col=0)
yTest_cat = pd.read_csv('CateyTest.csv', index_col=0)
yTrain_1 = yTrain_cat['1']
yTrain_5 = yTrain_cat['0.5']
yTrain_2 = yTrain_cat['0.2']
yTest_1 = yTest_cat['1']
yTest_5 = yTest_cat['0.5']
yTest_2 = yTest_cat['0.2']


# distribution = np.unique(yTrain_1, return_counts = True)
# print(distribution)
# perc_45 = distribution[1][3]/np.sum(distribution[1])
# print(perc_45)

# distribution = np.unique(yTest_5, return_counts = True)
# print(distribution)


# # lr
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error

# xTrain_lr=xTrain.drop(['Reviews', 'Category'], 1)
# xTest_lr=xTest.drop(['Reviews', 'Category'], 1)

# print(xTrain_lr.columns)

# reg = LinearRegression().fit(xTrain_lr, yTrain['numerical'])
# yHat = reg.predict(xTest_lr)
# error = mean_squared_error(yTest['numerical'], yHat)

# print(error)
# print(reg.coef_)

# Oversampling
ros = RandomOverSampler(random_state = 334)
xTrain_res, yTrain_res = ros.fit_resample(xTrain, yTrain_1)

# sm = SMOTE(random_state=334)
# xTrain_res, yTrain_res = sm.fit_resample(xTrain, yTrain_1)

# distribution = np.unique(yTrain_res, return_counts = True)
# print(distribution)

# from reg_resampler import resampler

# # Initialize the resampler object
# rs = resampler()
# df = np.append(xTrain, yTrain, axis=1)

# # You might recieve info about class merger for low sample classes
# # Generate classes
# y_classes = rs.fit(df, target=-1)
# # Create the actual target variable
# xTrain_res, yTrain_res = rs.resample(
#     sampler_obj=RandomOverSampler(random_state=27),
#     trainX=df,
#     trainY=y_classes
# )

# print(yTrain_res)
# print(xTrain_res)

# SVM
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

# # # Tune C, Kernel, shrinking
# grid = {
#     'C':            [0.1, 1, 10, 100, 1000],
#     'gamma':        [1, 0.1, 0.01, 0.001, 0.0001],
#     'kernel':       ['rbf']
#     # 'degree':       np.arange(1, 4, 1).tolist(),
#     # 'gamma':        ['scale', 'auto'],
#     # 'shrinking':    [True, False],
#     }

# svc = SVC()
# # clf = RandomizedSearchCV(estimator = svc,  param_distributions = grid , cv=5, n_jobs = -1, verbose = 2,  
# #                            random_state=42, n_iter=2)
# grid = GridSearchCV(estimator = svc, param_grid = grid, 
#                           cv = 5, refit = True, verbose = 2)
# grid.fit(xTrain_res, yTrain_res)
# print(grid.best_params_)
# print(grid.best_estimator_)

# # clf = RandomizedSearchCV(estimator = SVR(),  param_distributions = grid , cv=5, n_jobs = -1, verbose = 2,  
# #                            random_state=42, n_iter=2)
# grid = GridSearchCV(estimator = SVR(), param_grid = grid, 
#                           cv = 5, refit = True, verbose = 2)

# grid.fit(xTrain, np.ravel(yTrain))
# print(grid.best_params_)
# # print(grid.best_estimator_)

# svr = SVR(C = 0.1, gamma=0.1, kernel = 'rbf')
# svr.fit(xTrain, np.ravel(yTrain))
# yHat = svr.predict(xTest)
# acc = mean_squared_error(yHat, yTest)
# print("svr", acc)


# svm = SVC(C = 0.1, gamma=1, kernel = 'rbf')
# svm.fit(xTrain, yTrain_1)
# yHat = svm.predict(xTest)
# acc = accuracy_score(yHat, yTest_1)

# f1 = f1_score(yTest_1, yHat, average='weighted')
# print("svm", acc)
# print("svm f1", f1)

# print(svm.classes_)
# # computing precision, recall and f1 score for each class
# truth = np.zeros(4)
# predict = np.zeros(4)
# correct = np.zeros(4)
# classes = {'1-2': 0, '2-3': 1, '3-4': 2, '4-5': 3}
# for i in range(len(xTest)):
#     y = yTest_1[i]
#     predict[classes[yHat[i]]] += 1
#     truth[classes[yTrain_1[i]]] += 1
#     if (classes[yHat[i]] == classes[yTrain_1[i]]):
#         correct[classes[yHat[i]]] += 1

# # print(correct)
# # print(predict)
# # print(truth)

# precision = correct / predict
# recall = correct / truth
# f1 = 2 * (precision * recall) / (precision + recall)
# print(precision)
# print(recall)
# print(f1)

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


# # knn
# from sklearn.neighbors import KNeighborsClassifier

# knn = KNeighborsClassifier(metric = 'manhattan', n_neighbors = 1)
# knn.fit(xTrain, yTrain_1)
# yHat = knn.predict(xTest)
# acc = accuracy_score(yHat, yTest_1)
# f1 = f1_score(yTest_1, yHat, average='weighted')
# print("knn", acc)
# print("knn f1", f1)

# knn = KNeighborsClassifier(metric = 'manhattan', n_neighbors = 1)
# knn.fit(xTrain_res, yTrain_res)
# yHat = knn.predict(xTest)
# acc = accuracy_score(yHat, yTest_1)
# f1 = f1_score(yTest_1, yHat, average='weighted')
# print("knn res", acc)
# print("knn res f1", f1)

# from sklearn.linear_model import LogisticRegression

# lr = LogisticRegression(multi_class='ovr', solver='liblinear')
# lr.fit(xTrain, yTrain_1)
# yHat = lr.predict(xTest)
# acc = accuracy_score(yHat, yTest_1)
# f1 = f1_score(yTest_1, yHat, average='weighted')
# print("logreg", acc)
# print("logreg f1", f1)

# lr = LogisticRegression(multi_class='ovr', solver='liblinear')
# lr.fit(xTrain_res, yTrain_res)
# yHat = lr.predict(xTest)
# acc = accuracy_score(yHat, yTest_1)
# f1 = f1_score(yTest_1, yHat, average='weighted')
# print("logreg res", acc)
# print("logreg res f1", f1)
