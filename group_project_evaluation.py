import numpy as np 
import csv
import sys


from final_data_selection import data_selection
from polynomial_global import main_polynomial_function
from knn_global import main_knn_function
from knn_final_test import multi_knn



def main():

    file_name = sys.argv[1]
    data_selection(file_name)

    with open('final_training_data.csv', 'r') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        header = next(datareader)
        data = []
    

        for row in datareader:
            row_of_floats = list(map(float, row))
            data.append(row_of_floats)

        training_data_as_nparray = np.array(data)

    with open('final_test_data.csv', 'r') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        header = next(datareader)
        data = []
    

        for row in datareader:
            row_of_floats = list(map(float, row))
            data.append(row_of_floats)

        test_data_as_nparray = np.array(data)    




    best_polynomial_function = main_polynomial_function()        

    best_knn_paramaters = main_knn_function()

    best_knn_subset = best_knn_paramaters[0]
    optimum_k = best_knn_paramaters[2]


    knn_test_results = multi_knn(best_knn_subset,training_data_as_nparray,test_data_as_nparray,optimum_k+20)
    





    














