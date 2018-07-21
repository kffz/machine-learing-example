# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 11:20:41 2018

@author: 123
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cross_validation import train_test_split,StratifiedKFold,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import sklearn.tree as tree
from sklearn.externals.six import StringIO
from sklearn.ensemble import RandomForestClassifier

def plot_cv(cv,n_samples):
    masks = []
    for train,test in cv:
        mask = np.zeros(n_samples,dtype=bool)
        #print(mask)
        mask[test]=1
        masks.append(mask)
    plt.figure(figsize=(15,15))
    plt.imshow(masks,interpolation='none')
    plt.ylabel('Fold')
    plt.xlabel('Row #')

iris_data_clean = pd.read_csv('iris-data-clean.csv')
all_inputs = iris_data_clean[['sepal_length_cm','sepal_width_cm',
                              'petal_length_cm','petal_width_cm']].values
all_classes = iris_data_clean['class'].values
random_forest_classifier = RandomForestClassifier()
parameter_grid={'n_estimators':[5,10,25,50],
                'criterion':['gini','entropy'],
                'max_features':[1,2,3,4],
                'warm_start':[True,False]}
cross_validation = StratifiedKFold(all_classes,n_folds=10)
grid_search = GridSearchCV(random_forest_classifier,param_grid=parameter_grid,cv=cross_validation)
grid_search.fit(all_inputs,all_classes)
print('Best score:{}'.format(grid_search.best_score_))
print('Best parameters:{}'.format(grid_search.best_params_))
rf_classifier=grid_search.best_estimator_
rf_classifier_scores = cross_val_score(rf_classifier,all_inputs,all_classes,cv=10)
sb.boxplot(rf_classifier_scores)
sb.stripplot(rf_classifier_scores,jitter=True,color='white')
(training_inputs,testing_inputs,training_classes,testing_classes)=train_test_split(all_inputs,all_classes,train_size=0.75)
rf_classifier.fit(training_inputs,training_classes)
print(rf_classifier.predict(testing_inputs[:10]))
for input_features,prediction,actual in zip(testing_inputs[:10],
                                            rf_classifier.predict(testing_inputs[:10]),
                                            testing_classes[:10]):
    print('{}\t-->\t{}(Actual:{})'.format(input_features,prediction,actual))
