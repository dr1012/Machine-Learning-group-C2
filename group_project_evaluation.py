import numpy as np 
import csv
import sys

from linear_final_test import simple_linear_final_test
from simple_linear_regression  import simple_linear_main
from final_data_selection import data_selection
from polynomial_global import main_polynomial_function
from knn_final_test import multi_knn
from knn_global import main_knn_function
from poly_final_test import final_poly_test_function

from create_table import drawTable

import regression_rbf_kai
import descriptive_statistics

simple_linear_test_error = 0

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

    descriptive_statistics.mainStat()

    #input from best simple liner regression
    print("Linear Regression Learning...")
    simple_linear_train_error, simple_linear_test_error, simple_linear_weights =  simple_linear_main()
    print("simple linear  OK")


    #input from best polynomial regression
    print("Polynomial Learning Takes 1 min...")
    poly_parameter_list = main_polynomial_function()
    best_poly_subset = poly_parameter_list[0]
    best_degree = poly_parameter_list[2]

    print("poly parameter list " + str(poly_parameter_list))
    print(" polynomial  OK")


    #input from best KNN regression
    print("kNN Learning...")
    print("Takes around 10 mins...")
    best_knn_paramaters = main_knn_function()
    best_knn_subset = best_knn_paramaters[0]
    optimum_k = best_knn_paramaters[2]
    print("best knn subset" + str(type(best_knn_subset)))
    print("KNN OK")

#########################

    #testing RBF
    print("RBF Testing takes around 3 mins...")
    normalised_min_prediction, opt_reg, opt_scale, opt_centers = regression_rbf_kai.run_rbf_model()
    print("RBF min RMSE " + str(normalised_min_prediction))
    print("RBF OK")

    #testing KNN
    print("kNN Testing...")
    knn_test_results = multi_knn(best_knn_subset,training_data_as_nparray,test_data_as_nparray,int(optimum_k+20))
    print("final knn test (subset, min_error, k):  " + str(knn_test_results))

    #testing linear
    print("Linear Regression Testing...")
    simple_linear_result = simple_linear_final_test(simple_linear_weights,test_data_as_nparray)
    print("final simple linear test (test error): " + str(simple_linear_result))

    #testing polynomial
    print("Polynomial Testing...")
    poly_final_min_error, poly_final_degree = final_poly_test_function(best_poly_subset,test_data_as_nparray,best_degree)
    print("final polynomial test error and degree: " + str(poly_final_min_error) + "   " + str(poly_final_degree))

    # printing summary output
    headers = [" ","LR","Polynomial","RBF","kNN"]

    table = [["RMSE", float('%.3g'%simple_linear_result), float('%.3g'% poly_final_min_error),
              float('%.3g'%normalised_min_prediction), float('%.3g'%knn_test_results[1])]]

    headers2 = ["Linear Regression", "Polynomial Regression", "Radial Basis Function"]

    table2 = [["param-s: " + str(best_knn_paramaters),"subset: " + str(best_poly_subset),"reg param " + str(opt_reg)],
              ["subset: " + str(best_knn_subset),"degree: " + str(best_degree), "scale " + str(opt_scale)],
              ["k = " + str(optimum_k)," ","centres: " + str(opt_centers)]]

    # table2 = [["subset: " + str("0, 9, 1, 2, 7, 6"),"reg param = " + str("3.9810717055349694e-10"),"param-s: " + str("[10.0], 0.696993777439047, 54.0")],
    #           ["degree: " + str("2.0"), "scale " + str("206.913808111479"),"subset: " + str("10.0")],
    #           [" ","centres: " + str("0.05"),"k = " + str("54.0")]]

    print("Table1. Root Mean Square values for 4 models")
    drawTable(headers, table)
    print("\n")
    print("Table2. Parameter values for 4 models")
    drawTable(headers2, table2)


def export_simple_linear_regression_test_error():
    return simple_linear_test_error



if __name__ == '__main__':
    # this bit only runs when this script is called from the command line
    # but not when poly_fit_base.py is used as a library
    main()









