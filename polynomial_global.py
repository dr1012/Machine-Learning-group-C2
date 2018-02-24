import csv
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg

from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from poly_function_return import function_return




from polynomial_regression_individual import inidividual_poly_main_function
from poly_fit_multi_combinations import multi_poly_combinations

from poly_fit_plot import plot_train_test_errors

import itertools

# splitting the data 80:20 for training and testing the polynomial regression model


def main_polynomial_function():


    with open('final_training_data.csv', 'r') as csvfile:
            datareader = csv.reader(csvfile, delimiter=',')
            header = next(datareader)
            data = []
        

            for row in datareader:
                row_of_floats = list(map(float, row))
                data.append(row_of_floats)

    data_as_nparray = np.array(data)

    size  = len(data_as_nparray)
    train_size_float =  (0.8*size)

    train_size = int(train_size_float)





    internal_training_data_as_nparray = data_as_nparray[0:train_size]


    internal_test_data_as_nparray = data_as_nparray[train_size:]




    ############individual polynomial fits###################################


    individual_errors_arrays = inidividual_poly_main_function(internal_training_data_as_nparray,internal_test_data_as_nparray)


    npindividual_errors_arrays = np.array([individual_errors_arrays])

    npindividual_errors_arrays =  npindividual_errors_arrays.reshape((11,3))


    #sorted array by which attribute has the smallest error
    sorted_array = npindividual_errors_arrays[npindividual_errors_arrays[:,1].argsort()]


    # taking the  best 6 attributes as a subset
    subset_array = sorted_array[0:6]

    subset_attributes =  (subset_array[:][:,0:1]).flatten()



    ############testing all combinations of the subset of the best attributes ###################################
    subset_error_degree = []
    for L in range(0, len(subset_attributes)+1):
        for subset in itertools.combinations(subset_attributes, L):
    #every subset here is a tuple containing the indexes of the attributes in the subset
            if(len(subset)>1):
                print(subset)
                subset_list_temp = list(subset)
                subset_list =[int(i) for i in subset_list_temp]
                subset_error_degree.append(multi_poly_combinations(subset_list,internal_training_data_as_nparray,internal_test_data_as_nparray,(15-len(subset))))


    #subset, minimal error,  degree
    final_array = np.array(subset_error_degree)


    sorted_final_array = final_array[final_array[:,1].argsort()]


    if (subset_array[0][1])<=(sorted_final_array[0][1]):
        print("best polynomial atrtibute combination, error, degree: " + str(subset_array[0]))
        best_function = function_return(internal_training_data_as_nparray,internal_training_data_as_nparray[:][:,11:12],subset_array[0])
        return best_function

    else:
        print("best polynomial  atrtibute combination, error, degree: " + str(sorted_final_array[0]))
        best_function = function_return(internal_training_data_as_nparray,internal_training_data_as_nparray[:][:,11:12],sorted_final_array[0])
        return best_function













