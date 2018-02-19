import csv
import numpy as np
import numpy.random as random
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import operator
import math

from poly_fit_plot import plot_train_test_errors
from poly_fit_plot import plot_function
from poly_fit_plot import plot_function_and_data
from poly_fit_plot import plot_function_data_and_approximation

with open('training_data.csv', 'r') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        header = next(datareader)
        training_data = []
     

        for row in datareader:
            row_of_floats = list(map(float, row))
            training_data.append(row_of_floats)

# data is  of type list
training_data_as_array = np.array(training_data)



with open('test_data.csv', 'r') as csvfile1:
        datareader = csv.reader(csvfile1, delimiter=',')
        header = next(datareader)
        validation_data = []
     

        for row in datareader:
            row_of_floats = list(map(float, row))
            validation_data.append(row_of_floats)

# data is  of type list
test_data_as_array = np.array(validation_data)



def main():
    """
    This function contains example code that demonstrates how to use the 
    functions defined in poly_fit_base for fitting polynomial curves to data.
    """



def expand_to_monomials(inputs, degree):
    # create a list of the all inputs raise to each possible power
    expanded_inputs = []
    for i in range(degree+1):
        expanded_inputs.append(inputs**i)
    return np.array(expanded_inputs).transpose()

def create_prediction_function(degree, weights):
    """
    This function creates and returns a prediction function based on a
    feature mapping and some weights.

    The returned prediction function takes a set of input values and returns
    the predicted output for each.
    """
    # here is a function that is created on the fly from the input feature
    # mapping and weights
    def prediction_function(xs):
        expanded_xs = np.matrix(expand_to_monomials(xs, degree))
        ys = expanded_xs*np.matrix(weights).reshape((len(weights),1))
        return np.array(ys).flatten()
    # we return the function reference (handle) itself. This can be used like
    # any other function
    return prediction_function    

def least_squares_weights(processed_inputs, targets):
    """
    This method returns the weights that give the best linear fit between
    the processed inputs and the targets.
    """
    Phi = np.matrix(processed_inputs)
    targets = np.matrix(targets).reshape((len(targets),1))
    weights = linalg.inv(Phi.transpose()*Phi)*Phi.transpose()*targets
    return np.array(weights).flatten()

def regularised_least_squares_weights(
        processed_inputs, targets, reg_param):
    """
    This method returns the weights that give the best linear fit between
    the processed inputs and the targets penalised by some regularisation term
    (reg_param)
    """
    Phi = np.matrix(processed_inputs)
    targets = np.matrix(targets).reshape((len(targets),1))
    I = np.identity(Phi.shape[1])
    weights = linalg.inv(reg_param*I + Phi.transpose()*Phi)*Phi.transpose()*targets
    return np.array(weights).flatten()    


def root_mean_squared_error(y_true, y_pred):
    """
    Evaluate how closely predicted values (y_pred) match the true values
    (y_true, also known as targets)

    Parameters
    ----------
    y_true - the true targets
    y_pred - the predicted targets

    Returns
    -------
    mse - The root mean squared error between true and predicted target
    """
    N = len(y_true)
    # be careful, square must be done element-wise (hence conversion
    # to np.array)
    mse = np.sum((np.array(y_true).flatten() - np.array(y_pred).flatten())**2)/N
    return np.sqrt(mse)    


def train_and_test(
        degree, train_inputs, train_targets, test_inputs, test_targets,
        reg_param=None):
    """
    Fits a polynomial of degree "degree" to your training data then evaluates
    train and test errors and plots the resulting curves

    Parameters
    ----------
    degree - the degree of polynomial to fit
    train_inputs - the training inputs
    train_targets - the training targets
    test_inputs - the test inputs
    test_targets - the test targets
    reg_param (optional) - the regularisation strength. If not provided then
        the non-regularised least squares method is used.

    Returns
    -------
    train_error - the training error for the approximation
    test_error - the test error for the approximation
    """
    # convert both train and test inputs to monomial vectors
    processed_train_inputs = expand_to_monomials(train_inputs, degree)
    processed_test_inputs = expand_to_monomials(test_inputs, degree)
    # find the weights, using least squares or regularised least squares
    if reg_param is None:
        # use simple least squares approach
        weights = least_squares_weights(processed_train_inputs, train_targets)
    else:
        # use regularised least squares approach
        weights = regularised_least_squares_weights(
            processed_train_inputs, train_targets,  reg_param)
    # create the prediction function
    trained_func = create_prediction_function(degree, weights)
    # get the train and test errors and return them
    train_error = root_mean_squared_error(train_targets, trained_func(train_inputs))
    test_error = root_mean_squared_error(test_targets, trained_func(test_inputs))
    return train_error, test_error



def evaluate_degree(reg_param, degree_sequence=[0,1,2,3,4,5,6,7,8,9,10,11]):
    """
    Evaluates and plots test & train error (RMSE) for different degrees of
    polynomial fit to synthetic data.
    Allows one to essentially recreate Figure 1.5 from Bishop.

    Parameters
    ----------
    reg_param - the regularisation parameter if one is being used. Set to None
        if conventional least squares fit needed
    N - number of data points to sample for training
    degree_sequence - specifies which degrees of polynomial to fit to the data
    """
    # sample  train and test data
    train_inputs =  training_data_as_array[:,0:11]
    train_targets = training_data_as_array[:,11:12]
    test_inputs = validation_data_as_array[:,0:11]
    test_targets = validation_data_as_array[:,11:12]
    # 
    train_errors = []
    test_errors = []
    for degree in degree_sequence:
        # for each degree fit the data then evaluated train/test error
        train_error, test_error = train_and_test(
            degree, train_inputs, train_targets, test_inputs,
            test_targets, reg_param)
        train_errors.append(train_error)
        test_errors.append(test_error)
    # plot the results
    plot_train_test_errors("degree", degree_sequence, train_errors, test_errors)
    plt.show()

evaluate_degree(None)





if __name__ == '__main__':
    # this bit only runs when this script is called from the command line
    main()


