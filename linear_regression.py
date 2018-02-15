import csv
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
#import panda as pd



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
    
    degree = 1

    #retrieve train targets and inputs from data array
    targets = data_as_array[:,11]
    inputs = data_as_array[:,0:10]
#    test = data_as_array[:, [1,2,3]]
    
    # find the weights that fit the data in a least squares way
    ml_model_weights = ml_weights(inputs, targets)

    # find the regularised weights that fit the data in a least squares way
    reg_param = 1
    ml_model_regularised_weights = regularised_ml_weights(inputs, targets, reg_param)
                
    #get prediction (y-value) for the given weights (beta / coefficients)
    ys = linear_model_predict(inputs, ml_model_weights)
    ys_reg = linear_model_predict(inputs, ml_model_regularised_weights)

    #TRAINING: mean squared error
    mse = np.square(np.subtract(targets, ys)).mean()
    mse_regw = np.square(np.subtract(targets, ys_reg)).mean()
    print(mse)
    print(mse_regw)
    
    
    
    # sample data
    x = data_as_array[:,1]
    y = ys
    
    # fit with np.polyfit
    m, b = np.polyfit(x, y, 1)
    
    plt.plot(x, y, '.')
    plt.plot(x, m*x + b, '-')
    
    plt.show()
    
    
    
    
    #______________________________  TEST _____________________________________

    #retrieve targets and inputs from data array
    test_targets = test_data_as_array[:,11]
    test_inputs = test_data_as_array[:,0:10]

    #test: get prediction (y-value) for the given weights (beta / coefficients)
    test_ys = linear_model_predict(test_inputs, ml_model_weights)
    test_ys_reg = linear_model_predict(test_inputs, ml_model_regularised_weights)

    #test: mean squared error
    test_mse = np.square(np.subtract(test_targets, test_ys)).mean()
    test_mse_regw = np.square(np.subtract(test_targets, test_ys_reg)).mean()    

    print('test scores')
    print(test_mse)
    print(test_mse_regw)


#    fig, ax, hs = plot_function_data_and_approximation(linear_approx, inputs, targets)
    #4 true_func
    

    #plot the graph and data
   # ax.set_xticks([])
    #ax.set_yticks([])
    #fig.tight_layout()
    #fig.savefig("regression_linear.pdf", fmt="pdf")

    #plt.show()


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


def construct_polynomial_approx(degree, weights):
    """
    This function creates and returns a prediction function based on a
    feature mapping and some weights.

    The returned prediction function takes a set of input values and returns
    the predicted output for each.
    """
    # here is a function that is created on the fly from the input feature
    # mapping and weights
    def prediction_function(xs):
#        expanded_xs = np.matrix(expand_to_monomials(xs, degree))
        ys = xs*np.matrix(weights).reshape((len(weights),1))
        return np.array(ys).flatten()
    # we return the function reference (handle) itself. This can be used like
    # any other function
    return prediction_function

def expand_to_monomials(inputs, degree):
    """
    Create a design matrix from a 1d array of input values, where columns
    of the output are powers of the inputs from 0 to degree (inclusive)

    So if input is: inputs=np.array([x1, x2, x3])  and degree = 4 then
    output will be design matrix:
        np.array( [[  1.    x1**1   x1**2   x1**3   x1**4   ]
                   [  1.    x2**1   x2**2   x2**3   x2**4   ]
                   [  1.    x3**1   x3**2   x3**3   x3**4   ]])
    """
    expanded_inputs = []
    for i in range(degree+1):
        expanded_inputs.append(inputs**i)
    return np.array(expanded_inputs).transpose()


def plot_function_and_data(inputs, targets, markersize=5, **kwargs):

#true_func
    """
    Plot a function and some associated regression data in a given range

    parameters
    ----------
    inputs - the input data
    targets - the targets
    true_func - the function to plot
    markersize (optional) - the size of the markers in the plotted data
    <for other optional arguments see plot_function>

    returns
    -------
    fig - the figure object for the plot
    ax - the axes object for the plot
    lines - a list of the line objects on the plot
    """
#    fig, ax, lines = plot_function(true_func)
#    line, = ax.plot(inputs, targets, 'bo', markersize=markersize)
    line = ax.plot(inputs,targets)
#    lines.append(line)
    return fig, ax, line

def plot_function_data_and_approximation(
        predict_func, inputs, targets, linewidth=3, xlim=None,
        **kwargs):
    
    #4
    """
    Plot a function, some associated regression data and an approximation
    in a given range

    parameters
    ----------
    predict_func - the approximating function
    inputs - the input data
    targets - the targets
    true_func - the true function
    <for optional arguments see plot_function_and_data>

    returns
    -------
    fig - the figure object for the plot
    ax - the axes object for the plot
    lines - a list of the line objects on the plot
    """
    if xlim is None:
        xlim = (0,1)
    fig, ax, lines = plot_function_and_data(
        inputs, targets, linewidth=linewidth, xlim=xlim, **kwargs)
    #3 ,lines
    #3true_func
    xs = np.linspace(0, 1, 101)
    ys = predict_func(xs)
    line, = ax.plot(xs, ys, 'r-', linewidth=linewidth)
    lines.append(line)
    return fig, ax, lines






if __name__ == '__main__':
    # this bit only runs when this script is called from the command line
    main()
 