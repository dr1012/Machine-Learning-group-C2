import numpy as np
import csv
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import panda as pd


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 16:30:49 2018

@author: kai
"""

#load training data into matrix to be used
with open('training_data.csv', 'r') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        header = next(datareader)
        data = []
       
        for row in datareader:
            row_of_floats = list(map(float, row))
            data.append(row_of_floats)

        # data is  of type list
        data_as_array = np.array(data)
