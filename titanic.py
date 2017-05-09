#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 19:24:34 2017

@author: khanal
"""

import numpy as np
import pandas as pd
from random import shuffle

train_df = pd.read_csv('train.csv')
train_data = train_df.as_matrix()

test_df = pd.read_csv('test.csv')
test_df.insert(1, 'Survived', 0)
test_data = test_df.as_matrix()

def get_preprocessed_data(data):
    N, D = data.shape
    
    # pid = data[:,0].astype(float)
    survived = data[:,1].astype(float)
    pclass = data[:,2].astype(int)
    classes = np.zeros(shape=(N,3))
    sex = data[:,4]
    age = data[:,5]
    fare = data[:,-3].astype(float)
    
    nan_ages = 0
    for i in range(N):
        this_class = pclass[i]
        classes[i, this_class - 1] = 1
        if sex[i] == 'female':
            sex[i] = 1
        else:
            sex[i] = 0
        if str(age[i]) == 'nan':
            age[i] = 0
            nan_ages += 1
               
    # post conversion
    sex = sex.astype(float)
    age = age.astype(float)
    non_nan_ages = N - nan_ages
    avg_age = age.sum()/non_nan_ages
    
    for i in range(N):
        if age[i] == 0:
            age[i] = avg_age
               
    X = np.vstack((classes[:,0], classes[:,1], classes[:,2], sex, age, fare)).astype(float)    
    return X.T, survived
    
X_train, Y_train = get_preprocessed_data(train_data)
X_test, Y_test = get_preprocessed_data(test_data)

X_subtest = X_train[-100:]
Y_subtest = Y_train[-100:]

X_train = X_train[:-100]
Y_train = Y_train[:-100]

learning_rate = 0.01
costs = []

def sigmoid(z):
    return 1 / (1 + np.exp(-z))