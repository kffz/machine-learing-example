# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 14:26:23 2018

@author: 123
"""

import numpy as np
from sklearn.cross_validation import train_test_split,StratifiedKFold
a=np.arange(1000).reshape(200,5)
#print(a)
b=np.arange(200)
StratifiedKFold(b,n_folds=10)


