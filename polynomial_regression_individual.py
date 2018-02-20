import csv
import numpy as np
import matplotlib.pyplot as plt

# for creating synthetic data
from regression_samples import arbitrary_function_1
from regression_samples import arbitrary_function_2
from regression_samples import sample_data
# for performing regression
from regression_models import expand_to_monomials
from regression_models import ml_weights
from regression_models import construct_polynomial_approx
# for plotting results
from regression_plot import plot_function_data_and_approximation

from poly_fit_plot import plot_train_test_errors


with open('training_data.csv', 'r') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        header = next(datareader)
        training_data = []
     

        for row in datareader:
            row_of_floats = list(map(float, row))
            training_data.append(row_of_floats)

# data is  of type list
training_data_as_array = np.array(training_data)



with open('validation_data.csv', 'r') as csvfile1:
        datareader = csv.reader(csvfile1, delimiter=',')
        header = next(datareader)
        validation_data = []
     

        for row in datareader:
            row_of_floats = list(map(float, row))
            validation_data.append(row_of_floats)

# data is  of type list
validation_data_as_array = np.array(validation_data)





def main():
    """
    This function contains example code that demonstrates how to use the 
    functions defined in poly_fit_base for fitting polynomial curves to data.
    """
    # choose number of data-points and sample a pair of vectors: the input
    # values and the corresponding target values

    
    
    train_errors = []
    test_errors = []
    
    train_inputs = training_data_as_array[:][:,10:11]
    train_targets = training_data_as_array[:][:,11:12]

    test_inputs = validation_data_as_array[:][:,10:11]
    test_targets = validation_data_as_array[:][:,11:12]
 

 

       
  
    for x in range(0,11):
        train_errors = []
        test_errors = []
        degree_sequence=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

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


        plot_train_test_errors("degree", degree_sequence, train_errors, test_errors, header[x])
        plt.show()    
 



    

 














  

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



if __name__ == '__main__':
    # this bit only runs when this script is called from the command line
    # but not when poly_fit_base.py is used as a library
    main()
