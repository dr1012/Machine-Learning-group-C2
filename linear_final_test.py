import csv
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import numpy.linalg as linalg

def simple_linear_final_test(weights, test_data):


    test_inputs = test_data[:][:,0:11]
    test_targets = test_data[:][:,11:12]

    test_predicts = linear_model_predict(test_inputs, weights)


    test_error = root_mean_squared_error(test_targets, test_predicts)

    conf_low, conf_high, ste = conf_int(test_error)

    print("Linear Regression Confidence Interval: +-" + str(ste))

    return test_error, ste


def conf_int(error_array):


    ste = error_array/sqrt(1)

    conf_low = error_array - ste
    conf_high = error_array + ste

    return conf_low, conf_high, ste

def linear_model_predict(designmtx, weights):
    ys = np.matrix(designmtx)*np.matrix(weights).reshape((len(weights),1))
    return np.array(ys).flatten()

  

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
