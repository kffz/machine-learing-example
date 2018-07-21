# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 19:59:59 2018

@author: 123
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

iris_data = pd.read_csv('2.csv')
#print(iris_data.head())
#print(iris_data.describe())
#sb.pairplot(iris_data.dropna(),hue='class')
iris_data.loc[iris_data['class']=='versicolor','class']='Iris-versicolor'
iris_data.loc[iris_data['class']=='Iris-setossa','class']='Iris-setosa'
#print(iris_data['class'].unique())
iris_data = iris_data.loc[(iris_data['class']!='Iris-setosa')|(iris_data['sepal_width_cm']>=2.5)]
#iris_data.loc[iris_data['class']=='Iris-setosa','sepal_width_cm'].hist()
iris_data.loc[(iris_data['class']=='Iris-versicolor')&(iris_data['sepal_length_cm']<1.0)]
iris_data.loc[(iris_data['class']=='Iris-versicolor')&(iris_data['sepal_length_cm']<1.0),'sepal_length_cm']*=100
#iris_data.loc[iris_data['class']=='Iris-versicolor','sepal_length_cm'].hist()
iris_data.loc[(iris_data['sepal_length_cm'].isnull())|
        (iris_data['sepal_width_cm'].isnull())|
        (iris_data['petal_length_cm'].isnull())|
        (iris_data['petal_width_cm'].isnull())]
average_petal_width = iris_data.loc[iris_data['class']=='Iris-setosa','petal_width_cm'].mean()
iris_data.loc[(iris_data['class']=='Iris-setosa')&
              (iris_data['petal_width_cm'].isnull()),
              'petal_width_cm'] = average_petal_width
iris_data.loc[(iris_data['class']=='Iris-setosa')&
              (iris_data['petal_width_cm']==average_petal_width)]
iris_data.to_csv('iris-data-clean.csv',index=False)
iris_data_clean = pd.read_csv('iris-data-clean.csv')
#sb.pairplot(iris_data_clean,hue='class')
plt.figure(figsize=(10,10))
for column_index,column in enumerate(iris_data_clean.columns):
    if column == 'class':
        continue
    plt.subplot(2,2,column_index+1)
    sb.violinplot(x='class',y=column,data=iris_data_clean)