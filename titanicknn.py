#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 21:16:50 2017

@author: khanal
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

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
    nan_fares = 0
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
        if str(fare[i]) == 'nan':
            fare[i] = 0
            nan_fares += 1
               
    # post conversion
    sex = sex.astype(float)
    age = age.astype(float)
    non_nan_ages = N - nan_ages
    avg_age = age.sum()/non_nan_ages
    non_nan_fares = N - nan_fares
    avg_fare = fare.sum()/non_nan_fares
    
    for i in range(N):
        if age[i] == 0:
            age[i] = avg_age
        if fare[i] == 0:
            fare[i] = avg_fare
               
    age = (age - age.mean()) / age.std()
    fare = (fare - fare.mean()) / fare.std()
               
    X = np.vstack((classes[:,0], classes[:,1], classes[:,2], sex, age, fare)).astype(float)    
    #X = np.vstack((classes[:,0], classes[:,1], classes[:,2], sex, age)).astype(float)    
    return X.T, survived
    
X_train, Y_train = get_preprocessed_data(train_data)
X_test, Y_test = get_preprocessed_data(test_data)

N, D = X_train.shape
N1, D1 = X_test.shape

X = np.ones((N, D+1))
X[:,1:] = X_train
X_train = X

X = np.ones((N1, D1+1))
X[:,1:] = X_test
X_test = X

X_subtest = X_train[-100:]
Y_subtest = Y_train[-100:]

X_train = X_train[:-100]
Y_train = Y_train[:-100]

W = np.random.randn(D+1) / np.sqrt(D+1)


learning_rate = 0.001
epochs = 100000
costs = []
r1 = 10
r2 = 0.1

result = X_test.dot(X_train.T)
result_hat = np.zeros((100,791))

for i in range(X_subtest.shape[0]):
    for j in range(X_train.shape[0]):
        num = X_train[j] - X_subtest[i]
        result_hat[i,j] = np.sqrt(num.dot(num.T))

Y_test = np.zeros(X_test.shape[0])
Y_hat = np.zeros(100)

k = 3

for i in range(result_hat.shape[0]):
    indices_hat = result_hat[i].argsort()
    k_vals_hat = Y_train[indices][0:k].astype(int)
    value_hat = np.bincount(k_vals_hat)
    if len(value_hat) == 1 or value_hat[0] > value_hat[1]:
        Y_hat[i] = 0
    else:
        Y_hat[i] = 1
             
print((Y_hat == Y_subtest).mean())


for i in range(result.shape[0]):
    indices = result[i].argsort()
    k_vals = Y_train[indices][0:k].astype(int)
    value = np.bincount(k_vals)
    if len(value) == 1 or value[0] > value[1]:
        Y_test[i] = 0
    else:
        Y_test[i] = 1
              
    


#with open('result.csv', 'w+') as out_file:
#    for i in range(len(Y)):
#        out_file.write('{0},{1}\n'.format(Y[i,0], Y[i,1]))