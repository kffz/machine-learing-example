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
#print(all_inputs)
#(training_inputs,testing_inputs,training_classes,testing_classes)=train_test_split(all_inputs,all_classes,train_size=0.75,random_state=1)
#decision_tree_classifier = DecisionTreeClassifier()
#decision_tree_classifier.fit(training_inputs,training_classes)
#decision_tree_classifier.score(testing_inputs,testing_classes)
#model_accuracies=[]
#for repetition in range(1000):
#    (training_inputs,testing_inputs,training_classes,testing_classes)=train_test_split(all_inputs,all_classes,train_size=0.75)
#    decision_tree_classifier = DecisionTreeClassifier()
#    decision_tree_classifier.fit(training_inputs,training_classes)
#    classifier_accuracy=decision_tree_classifier.score(testing_inputs,testing_classes)
#    model_accuracies.append(classifier_accuracy)
#sb.distplot(model_accuracies)
#print(type(StratifiedKFold(all_classes,n_folds=10)))
#plot_cv(StratifiedKFold(all_classes,n_folds=10),len(all_classes))
decision_tree_classifier = DecisionTreeClassifier()
parameter_grid = {'criterion':['gini','entropy'],'splitter':['best','random'],'max_depth':[1,2,3,4,5],'max_features':[1,2,3,4]}
cross_validation = StratifiedKFold(all_classes,n_folds=10)
#cv_scores = cross_val_score(decision_tree_classifier,all_inputs,all_classes,cv=10)
grid_search = GridSearchCV(decision_tree_classifier,param_grid=parameter_grid,cv=cross_validation)
grid_search.fit(all_inputs,all_classes)
print('Best score:{}'.format(grid_search.best_score_))
print('Best parameters:{}'.format(grid_search.best_params_))
#grid_visualization=[]
#print(type(grid_search.cv_results_))
#for grid_pair in grid_search.grid_scores_:
#    grid_visualization.append(grid_pair.mean_validation_score)
#grid_visualization = np.array(grid_visualization)
#grid_visualization.shape=(5,4)
#sb.heatmap(grid_visualization,cmap='Blues')
#print(grid_search.param_grid['max_features'])
#plt.xticks(np.arange(4)+0.5,grid_search.param_grid['max_features'])
#plt.yticks(np.arange(5)+0.5,grid_search.param_grid['max_depth'][::-1])
#plt.xlabel('max_features')
#plt.ylabel('max_depth')