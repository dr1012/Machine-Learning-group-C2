import csv
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

import statsmodels.formula.api as sm

# for performing regression
from regression_models import construct_rbf_feature_mapping
from regression_models import construct_feature_mapping_approx
# for plotting results
from regression_plot import plot_train_test_errors
# for evaluating fit
from regression_train_test import train_and_test
# two new functions for cross validation
from regression_train_test import create_cv_folds
from regression_train_test import cv_evaluation_linear_model

#importing methods for linear regression model coefficient estimations



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
        
        
#to test how well the cross validation works with the entire data set
with open('winequality-red-commas.csv', 'r') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        header = next(datareader)
        data = []
        
        for row in datareader:
            row_of_floats = list(map(float, row))
            data.append(row_of_floats)

        # data is  of type list

        cross_val_dataset = np.array(data)        


def main():
    """
    This function contains example code that demonstrates how to use the 
    functions defined in poly_fit_base for fitting polynomial curves to data.
    """

    #selected features: volatile acidity, citric acid, chlorides, density, pH, 
    # sulphates, alcohol

    #statsmodels method to fit a regression model to selected features

#    df = pd.DataFrame({"Y": data_as_array[:,11], "A": data_as_array[:,10], "B": data_as_array[:,1] , "C": data_as_array[:,2]
#    , "D": data_as_array[:,4], "E": data_as_array[:,7], "F": data_as_array[:,8], "G": data_as_array[:,9]})
#    result = sm.ols(formula="Y ~ A + B + C + D + E + F + G", data=df).fit()
#    print(result.params)
#    print(result.summary())

    
    
    #manual version of calculating RMSE
    degree = 1

    #retrieve train targets and inputs from data array
    targets = cross_val_dataset[:,11]    
    inputs = cross_val_dataset[:, [2,6,7,8,9]]
    
    # get the cross-validation folds
    N = len(cross_val_dataset)
    num_folds = 5
    folds = create_cv_folds(N, num_folds)
    
    print(folds)

    #evaluate then plot the performance of different coefficient estimates    
    evaluate_linReg_weights(inputs, targets,folds)
    
    
    
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
    mse_reg = np.square(np.subtract(targets, ys_reg)).sum()

    #root mean squared error (to make comparison valid across differently sized data sets)
    Rmse = np.sqrt(2*mse/len(data_as_array))

    
    
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


def evaluate_linReg_weights(inputs, targets, folds, reg_param=None):
    """
    evaluate then plot the performance of different weight coefficients
    """
    
     # fix the reg_param
#    reg_param = 0.08
#    # fix the scale
#    scale = 0.03

    # choose a range of numbers of centres
#    if default_linReg_weights is None:
#        default_linReg_weights = np.arange(-1000,1000)
#    num_values = folds.size
    num_folds = len(folds)
    coefficientindices = []
    
    # create some arrays to store results
    train_mean_errors = np.zeros(num_folds)
    test_mean_errors = np.zeros(num_folds)
    train_stdev_errors = np.zeros(num_folds)
    test_stdev_errors = np.zeros(num_folds)

    # run the experiments
    for w in range(num_folds):
        
#        weights = np.linspace(0,1,lin_weights)

#        feature_mapping = construct_rbf_feature_mapping(weights,scale)
        designmtx = inputs
        # r is the index of reg_param, reg_param is the regularisation parameter
        # cross validate with this regularisation parameter
        train_errors, test_errors = cv_evaluation_linear_model(
            designmtx, targets, folds)
        
        print('________________________________________________')
        print(train_errors)        
        print(test_errors)
        print('________________________________________________')
        # we're interested in the average (mean) training and testing errors
        train_mean_error = np.mean(train_errors)
        test_mean_error = np.mean(test_errors)
        train_stdev_error = np.std(train_errors)
        test_stdev_error = np.std(test_errors)
        # store the results
        train_mean_errors[w] = train_mean_error
        test_mean_errors[w] = test_mean_error
        train_stdev_errors[w] = train_stdev_error
        test_stdev_errors[w] = test_stdev_error
        
#        print('testNUUUUUUM__________________________________________')
#        
        coefficientindices.append(w)
#        print(coefficientindices)
#    print(len(coefficientindices))
        
    print(train_mean_errors)
    print(test_mean_errors)
    print(train_stdev_errors)
    print(test_stdev_errors)
    
        

    # Now plot the results
    fig, ax = plot_train_test_errors(
        "regression weight index", coefficientindices, train_errors, test_errors)
    # Here we plot the error ranges too: mean plus/minus 1 standard error.
    # 1 standard error is the standard deviation divided by sqrt(n) where
    # n is the number of samples. 
    # (There are other choices for error bars.)

    # using the train_errors instead of train_mean_errors because the values are all the same
    #Hypothesis: the underlying coefficients don't change between the fitting runs in the for loop and therefore all 5 runs produce the same result

    # TRAIN error bars
#    lower = train_mean_errors - train_stdev_errors/np.sqrt(num_folds)
    lower = train_errors - train_stdev_errors/np.sqrt(num_folds)
#    upper = train_mean_errors + train_stdev_errors/np.sqrt(num_folds)
    upper = train_errors + train_stdev_errors/np.sqrt(num_folds)
    ax.fill_between(coefficientindices, lower, upper, alpha=0.2, color='b')
    ax.set_ylim([.7,.8])

#    ax.set_xticklabels([1,2,3,4,5])
    
    # TEST error bars
#    lower = test_mean_errors - test_stdev_errors/np.sqrt(num_folds)
    lower = test_errors - test_stdev_errors/np.sqrt(num_folds)
#    upper = test_mean_errors + test_stdev_errors/np.sqrt(num_folds)
    upper = test_errors + test_stdev_errors/np.sqrt(num_folds)
    ax.fill_between(coefficientindices, lower, upper, alpha=0.2, color='r')
    
    
 

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

