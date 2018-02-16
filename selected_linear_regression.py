import csv
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

import statsmodels.formula.api as sm


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 17:18:49 2018

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


with open('test_data.csv', 'r') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        header = next(datareader)
        data = []
        
        for row in datareader:
            row_of_floats = list(map(float, row))
            data.append(row_of_floats)

        # data is  of type list
        test_data_as_array = np.array(data)


def main():
    """
    This function contains example code that demonstrates how to use the 
    functions defined in poly_fit_base for fitting polynomial curves to data.
    """

    #selected features: volatile acidity, citric acid, chlorides, density, pH, 
    # sulphates, alcohol

    #statsmodels method to fit a regression model to selected features
    df = pd.DataFrame({"Y": data_as_array[:,11], "A": data_as_array[:,10], "B": data_as_array[:,1] , "C": data_as_array[:,2]
    , "D": data_as_array[:,4], "E": data_as_array[:,7], "F": data_as_array[:,8], "G": data_as_array[:,9]})
    result = sm.ols(formula="Y ~ A + B + C + D + E + F + G", data=df).fit()
    print(result.params)
    print(result.summary())

    
    #manual version of calculating RMSE
    degree = 1

    #retrieve train targets and inputs from data array
    targets = data_as_array[:,11]    
    inputs = data_as_array[:, [1,2,4,7,8,9,10]]
    
    # find the weights that fit the data in a least squares way
    ml_model_weights = ml_weights(inputs, targets)

    # find the regularised weights that fit the data in a least squares way
    reg_param = 1
    ml_model_regularised_weights = regularised_ml_weights(inputs, targets, reg_param)
                
    #get prediction (y-value) for the given weights (beta / coefficients)
    ys = linear_model_predict(inputs, ml_model_weights)
    ys_reg = linear_model_predict(inputs, ml_model_regularised_weights)

    #TRAINING: sum of squared error
    mse = np.square(np.subtract(targets, ys)).sum()
    mse_regw = np.square(np.subtract(targets, ys_reg)).sum()

    #root mean squared error (to make comparison valid across differently sized data sets)
    Rmse = np.sqrt(2*mse/len(data_as_array))
    print(Rmse)
    
    
    #______________________________  TEST _____________________________________

    #retrieve targets and inputs from data array
    test_targets = test_data_as_array[:,11]
    test_inputs = test_data_as_array[:, [1,2,4,7,8,9,10]]

    #test: get prediction (y-value) for the given weights (beta / coefficients)
    test_ys = linear_model_predict(test_inputs, ml_model_weights)
    test_ys_reg = linear_model_predict(test_inputs, ml_model_regularised_weights)

    #test: sum of squared error
    test_mse = np.square(np.subtract(test_targets, test_ys)).sum()
    test_mse_regw = np.square(np.subtract(test_targets, test_ys_reg)).sum()    
    
    #Root mean square error
    test_Rmse = np.sqrt(2*test_mse/len(test_data_as_array))

    print(test_mse)
    print(test_Rmse)
    


# Polynomial Regression
def polyfit(x, y, degree):
    results = {}

    coeffs = np.polyfit(x, y, degree)

     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot
    
    print('hello, world')
    return results



def ml_weights(inputmtx, targets):
    """
    This method returns the weights that give the best linear fit between
    the processed inputs and the targets.
    """
    Phi = np.matrix(inputmtx)
    targets = np.matrix(targets).reshape((len(targets),1))
    weights = linalg.inv(Phi.transpose()*Phi)*Phi.transpose()*targets
    return np.array(weights).flatten()

def regularised_ml_weights(
        inputmtx, targets, reg_param):
    """
    This method returns the weights that give the best linear fit between
    the processed inputs and the targets penalised by some regularisation term
    (reg_param)
    """
    Phi = np.matrix(inputmtx)
    targets = np.matrix(targets).reshape((len(targets),1))
    I = np.identity(Phi.shape[1])
    weights = linalg.inv(reg_param*I + Phi.transpose()*Phi)*Phi.transpose()*targets
    return np.array(weights).flatten()


def linear_model_predict(designmtx, weights):
    ys = np.matrix(designmtx)*np.matrix(weights).reshape((len(weights),1))
    return np.array(ys).flatten()



if __name__ == '__main__':
    # this bit only runs when this script is called from the command line
    main()

