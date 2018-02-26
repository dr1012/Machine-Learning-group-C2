import csv
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter



# for plotting results
from regression_plot import plot_function_data_and_approximation

from poly_fit_plot import plot_train_test_errors





from expand_to_monomials_methods import expand_to_monomials_2
from expand_to_monomials_methods import expand_to_monomials_3
from expand_to_monomials_methods import expand_to_monomials_4
from expand_to_monomials_methods import expand_to_monomials_5
from expand_to_monomials_methods import expand_to_monomials_6
from expand_to_monomials_methods import expand_to_monomials_7
from expand_to_monomials_methods import expand_to_monomials_8
from expand_to_monomials_methods import expand_to_monomials_9
from expand_to_monomials_methods import expand_to_monomials_10



def ml_weights(inputmtx, targets):
   
    Phi = np.matrix(inputmtx)
    targets = np.matrix(targets).reshape((len(targets),1))
    weights = linalg.inv(Phi.transpose()*Phi)*Phi.transpose()*targets
    return np.array(weights).flatten()

def construct_polynomial_approx(length,degree,weights):

    
   
    def prediction_function(xs):
        expanded_xs = np.matrix(eval("expand_to_monomials_"+str(length)+"(xs,"+str(degree)+")"))
        ys = expanded_xs*np.matrix(weights).reshape((len(weights),1))
        return np.array(ys).flatten()
   
    return prediction_function    


def multi_poly_combinations(subset,training_data, test_data, max_degree):
 
    #subset = list of the indexes of the attributes. i.e subset = [0,1,5,6]

    
    
    train_errors = []
    test_errors = []
    

    train_targets = training_data[:][:,11:12]

    test_inputs = test_data[:][:,subset]
    test_targets = test_data[:][:,11:12]
 

    length = len(subset)

    degree_sequence  = range(1,max_degree)

    training_subset = training_data[:][:,subset]


    test_degree_error_pairs = []

    for degree in degree_sequence:

        processed_inputs = eval("expand_to_monomials_"+str(length)+"(training_subset, degree)")

        weights = ml_weights(processed_inputs, train_targets)

        approx_func = construct_polynomial_approx(length,degree,weights)

        train_error = root_mean_squared_error(train_targets, approx_func(training_subset))
        test_error = root_mean_squared_error(test_targets, approx_func(test_inputs))

        train_errors.append(train_error)
        test_errors.append(test_error)
        test_degree_error_pairs.append([degree, test_error])

    
    npTestErrors  = np.array(test_degree_error_pairs)
    min_error = np.min(npTestErrors[:,1:2])
    min_index = np.argmin(npTestErrors[:,1:2])
    degree_min_error = npTestErrors[min_index][0]
    
    
    
    title =  str(length)+"-D Poly regression, subset: " + str(subset) + " min rms error: " + str(min_error) + " at degree = " + str(degree_min_error)
    plot_train_test_errors("degree", degree_sequence, train_errors, test_errors,title)
    plt.savefig("n-D_poly_"+str(subset)+".pdf", fmt="pdf")

    
    return [subset, min_error, degree_min_error]
    
    




     


  

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



