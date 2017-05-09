#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 19:24:34 2017

@author: khanal
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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
    #X = np.vstack((sex, age, fare)).astype(float)    
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


learning_rate = 0.00001
epochs = 1000000
costs = []
r1 = 0.1

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cross_entropy(y, t):
    E = 0
    for i in range(len(t)):
        if t[i] == 1:
            E -= np.log(y[i])
        else:
            E -= np.log(1-y[i])
    return E

def ReLu(z):
    return np.log(1 + np.exp(z))

def feed_forward(w, x):
    return x.dot(w)

for e in range(epochs):
    Y_hat = (sigmoid(feed_forward(W, X_train)))
    delta = Y_train - Y_hat
    #err = cross_entropy(Y_hat, Y_train)
    #if e % (epochs / 100) == 0:
    #    costs.append(err)
    W += learning_rate*(X_train.T.dot(delta) - W*r1)
    
#plt.plot(costs)
#plt.show()

Y_hat = sigmoid(feed_forward(W, X_train)).round()
res = ((Y_hat == Y_train))
print(res.mean())

Y_hat = sigmoid(feed_forward(W, X_subtest)).round()
res = ((Y_hat == Y_subtest))
print(res.mean())

Y_hat = sigmoid(feed_forward(W, X_test)).round().astype(int).astype(str)
Y1 = test_data[:,0].astype(str)
Y = np.zeros((N1+1, 2)).astype(str)
Y[1:] = np.vstack((Y1, Y_hat)).T.astype(str)
Y[0] = np.array(['PassengerId','Survived'])

with open('result.csv', 'w+') as out_file:
    for i in range(len(Y)):
        out_file.write('{0},{1}\n'.format(Y[i,0], Y[i,1]))