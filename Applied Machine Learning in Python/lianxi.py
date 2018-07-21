# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 14:08:07 2018

@author: 123
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

#创建X特征和y响应
iris = load_iris()
X=iris.data
y=iris.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=4)

#使用默认参数实例化模型
def logregression(X_train,X_test,y_train,y_test):
    logreg = LogisticRegression()
    logreg.fit(X_train,y_train)
    y_pred=logreg.predict(X_test)
    print('logisticregression:',metrics.accuracy_score(y_test,y_pred))
    
def KnnClassifier5(X_train,X_test,y_train,y_test):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    print('KNeighborsClassifier5:',metrics.accuracy_score(y_test,y_pred))
    
def KnnClassifier1(X_train,X_test,y_train,y_test):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    print('KNeighborsClassifier1:',metrics.accuracy_score(y_test,y_pred))

def KnnClassifier(X_train,X_test,y_train,y_test):
    k_range = list(range(1,26))
    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train,y_train)
        y_pred = knn.predict(X_test)
        scores.append(metrics.accuracy_score(y_test,y_pred))
    plt.plot(k_range,scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Testing Accuracy')

KnnClassifier(X_train,X_test,y_train,y_test)        
