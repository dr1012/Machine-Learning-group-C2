import csv
import numpy as np
import matplotlib.pyplot as plt


from regression_models import expand_to_monomials
from regression_models import ml_weights
from regression_models import construct_polynomial_approx
# for plotting results
from regression_plot import plot_function_data_and_approximation

from poly_fit_plot import plot_train_test_errors

# Hiding warning about too many figures
plt.rcParams.update({'figure.max_open_warning': 0})

with open('winequality-red.csv', 'r') as csvfile:
        datareader = csv.reader(csvfile, delimiter=';')
        header = next(datareader)




def inidividual_poly_main_function(training_data_as_array, validation_data_as_array):

    

 

 
    total_minErrors_and_optimal_degrees = []

  
    for x in range(0,11):
        train_errors = []
        test_errors = []

        test_degree_error_pairs = []

        degree_sequence=range(1,14)

        train_inputs = training_data_as_array[:][:,x:x+1]
        train_targets = training_data_as_array[:][:,11:12]

        test_inputs = validation_data_as_array[:][:,x:x+1]
        test_targets = validation_data_as_array[:][:,11:12]

       

        for degree in degree_sequence:
            processed_inputs = expand_to_monomials(train_inputs, degree)

            weights = ml_weights(processed_inputs, train_targets)

            approx_func = construct_polynomial_approx(degree, weights)

            train_error = root_mean_squared_error(train_targets, approx_func(train_inputs))
            test_error = root_mean_squared_error(test_targets, approx_func(test_inputs))

            train_errors.append(train_error)
            test_errors.append(test_error)

            test_degree_error_pairs.append([degree, test_error])


        
        npTestErrors  = np.array(test_degree_error_pairs)
        min_error = np.min(npTestErrors[:,1:2])
        min_index = np.argmin(npTestErrors[:,1:2])
        degree_min_error = npTestErrors[min_index][0]

        
        total_minErrors_and_optimal_degrees.append([float(x), min_error, float(degree_min_error)])

        

        title =  "poly fit of " + str(header[x]) + " min rms error: " + str(min_error) + " at degree = " + str(degree_min_error)
        plot_train_test_errors("degree", degree_sequence, train_errors, test_errors, title)
        plt.savefig("1-D_poly_"+header[x]+".pdf", fmt="pdf")
        plt.close()
        

    return total_minErrors_and_optimal_degrees





    

 














  

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


