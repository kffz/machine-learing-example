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

decision_tree_classifier = DecisionTreeClassifier()
parameter_grid = {'criterion':['gini','entropy'],'splitter':['best','random'],'max_depth':[1,2,3,4,5],'max_features':[1,2,3,4]}
cross_validation = StratifiedKFold(all_classes,n_folds=10)
grid_search = GridSearchCV(decision_tree_classifier,param_grid=parameter_grid,cv=cross_validation)
grid_search.fit(all_inputs,all_classes)
print('Best score:{}'.format(grid_search.best_score_))
print('Best parameters:{}'.format(grid_search.best_params_))
decesion_tree_classifier = grid_search.best_estimator_
decision_tree_classifier.fit(all_inputs,all_classes)
#with open('iris.dot','w') as out_file:
#    out_file = tree.export_graphviz(decision_tree_classifier,out_file=out_file)
dt_scores = cross_val_score(decision_tree_classifier,all_inputs,all_classes,cv=10)
sb.boxplot(dt_scores)
sb.stripplot(dt_scores,jitter=True,color='white')
