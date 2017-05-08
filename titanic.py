#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 19:24:34 2017

@author: khanal
"""

import numpy as np
import pandas as pd

df = pd.read_csv('train.csv')
data = df.as_matrix()

N, D = data.shape

pid = data[:,0].astype(float)
survived = data[:,1].astype(float)
pclass = data[:,2].astype(int)
classes = np.zeros(shape=(N,3))
sex = data[:,4]

for i in range(N):
    this_class = pclass[i]
    classes[i, this_class - 1] = 1
    if sex[i] == 'female':
        sex[i] = 1
    else:
        sex[i] = 0
           
# post conversion
sex = sex.astype(float)
           