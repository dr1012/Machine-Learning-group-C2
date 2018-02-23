import numpy as np
import csv
import matplotlib.pyplot as plt

from regression_models import construct_rbf_feature_mapping

from regression_train_test import train_and_test_split
from regression_train_test import train_and_test_partition
from regression_train_test import train_and_test
from regression_train_test import simple_evaluation_linear_model
from regression_train_test import cv_evaluation_linear_model
from regression_train_test import create_cv_folds

from regression_plot import exploratory_plots
from regression_plot import plot_train_test_errors


#with open('winequality-red-commas.csv', 'r') as csvfile:
#        datareader = csv.reader(csvfile, delimiter=',')
#        header = next(datareader)
#        data = []
#       
#        for row in datareader:
#            row_of_floats = list(map(float, row))
#            data.append(row_of_floats)
#
#        # data is  of type list
#        data_as_array = np.array(data)


def main(ifname, delimiter=",", columns=None, has_header=True,
        test_fraction=0.25):
    """
    To be called when the script is run. This function creates, fits and plots
    synthetic data, and then fits and plots imported data (if a filename is
    provided). In both cases, data is 2 dimensional real valued data and is fit
    with maximum likelihood 2d gaussian.

    parameters
    ----------
    ifname -- filename/path of data file. 
    delimiter -- delimiter of data values
    has_header -- does the data-file have a header line
    columns -- a list of integers specifying which columns of the file to import
        (counting from 0)    
    """
    # if no file name is provided then use synthetic data
    data, field_names = import_data(
            ifname, delimiter=delimiter, has_header=has_header, columns=columns)
    exploratory_plots(data, field_names)
    N = data.shape[0]    
    inputs = data[:,0:11]
    targets = data[:,-1]

    
    
#    train_error_linear, test_error_linear = evaluate_linear_approx(inputs, targets, test_fraction)
#    evaluate_rbf_for_various_reg_params(inputs, targets, test_fraction, test_error_linear)
    
    
    parameter_search_rbf(inputs, targets, test_fraction)
    plt.show()

#
#def evaluate_linear_approx(inputs, targets, test_fraction):
#    # the linear performance
#    train_error, test_error = simple_evaluation_linear_model(
#        inputs, targets, test_fraction=test_fraction)
#    print("Linear Regression:")
#    print("\t(train_error, test_error) = %r" % ((train_error, test_error),))
#    return train_error, test_error
#
#def evaluate_rbf_for_various_reg_params(
#        inputs, targets, test_fraction, test_error_linear):
#    """
#    """
#
#    # for rbf feature mappings
#    # for the centres of the basis functions choose 10% of the data
#    N = inputs.shape[0]
#
#    #centres
#    centres = inputs[np.random.choice([False,True], size=N, p=[0.9,0.1]),:]
#    print(centres)
#    print("centres.shape = %r" % (centres.shape,))
#    
#    #scale
#    scale = 10. # of the basis functions
#
#    feature_mapping = construct_rbf_feature_mapping(centres,scale)
#    designmtx = feature_mapping(inputs)
#    train_part, test_part = train_and_test_split(N, test_fraction=test_fraction)
#    train_designmtx, train_targets, test_designmtx, test_targets = \
#        train_and_test_partition(
#            designmtx, targets, train_part, test_part)
#
#    # output the shapes of the train and test parts for debugging
#    print("train_designmtx.shape = %r" % (train_designmtx.shape,))
#    print("test_designmtx.shape = %r" % (test_designmtx.shape,))
#    print("train_targets.shape = %r" % (train_targets.shape,))
#    print("test_targets.shape = %r" % (test_targets.shape,))
#
#    # the rbf feature mapping performance
#    reg_params = np.logspace(-15,-1, 11)
#    train_errors = []
#    test_errors = []
#    
#    for reg_param in reg_params:
#        print("Evaluating reg_para " + str(reg_param))
#        train_error, test_error = simple_evaluation_linear_model(
#            designmtx, targets, test_fraction=test_fraction, reg_param=reg_param)
#        train_errors.append(train_error)
#        test_errors.append(test_error)
#
#    #plot the results
#    fig , ax = plot_train_test_errors(
#        "$\lambda$", reg_params, train_errors, test_errors)
#    # we also want to plot a straight line showing the linear performance
#    xlim = ax.get_xlim()
#    ax.plot(xlim, test_error_linear*np.ones(2), 'g:')
#    ax.set_title("test test")
#    ax.set_xscale('log')


def parameter_search_rbf(inputs, targets, test_fraction):
    """
    """
    N = inputs.shape[0]
    folds_num = 5
    center_nums = 5
    
    
    #selection of centre_proportions to be varied     
    sample_fractions = np.array([0.01,0.05,0.1,0.15,0.2])    

     #normalise input values
    for i in range(inputs.shape[1]):
        inputs[:,i] = ((inputs[:,i] - np.mean(inputs[:,i]))/  np.std(inputs[:,i]))
    
    #parameters to be optimised
    scales = np.logspace(0,4, 20) # of the basis functions
    print('scales: %r' % scales)
    
    reg_params = np.logspace(-15,-1, 11) # choices of regularisation strength
    print('reg_params: %r' % reg_params)
    
    #create folds to run cross-validation on the parameters to be optimised
    folds = create_cv_folds(N,folds_num)
 
    # create empty 3d arrays to store the train and test errors
    train_errors = np.empty((sample_fractions.size, scales.size,reg_params.size))
    test_errors = np.empty((sample_fractions.size, scales.size,reg_params.size))
          
    #store the location of the optimal value in the error matrix
    optimal_h = 0    
    optimal_i = 0
    optimal_j = 0
    
    min_test_error = 10*10
    
    #iterate over different number of centres
    for h, sample_fraction in enumerate(sample_fractions):
        
        #determine a different number of centres and thereby their locations
        p = (1-sample_fraction,sample_fraction)
        centres = inputs[np.random.choice([False,True], size=N, p=p),:]
    
        # iterate over the scales
        for i,scale in enumerate(scales):
            # i is the index, scale is the corresponding scale
            # we must recreate the feature mapping each time for different scales
            feature_mapping = construct_rbf_feature_mapping(centres,scale)
            designmtx = feature_mapping(inputs)

            # iteratre over the regularisation parameters
            for j, reg_param in enumerate(reg_params):
                # j is the index, reg_param is the corresponding regularisation
                # parameter
                # train and test the data
            
                #array of k-error values 
                train_error, test_error, weight = cv_evaluation_linear_model(designmtx, targets, folds,reg_param = reg_param )
                
                # store the train and test errors in our 3d arrays
                train_errors[h,i,j] = np.mean(train_error)
                test_errors[h,i,j] = np.mean(test_error)
                
                if(np.mean(test_error) < min_test_error):
                    min_test_error = np.mean(test_error)
                    optimal_h = h
                    optimal_i = i
                    optimal_j = j
                    
    print("optimal test_error = %r" %min_test_error,"optimal scale = %r" %scales[optimal_i], "optimal centres: %r" % sample_fractions[optimal_h] )


#    print("Best joint choice of parameters:")
##    print("\tscale %.2g and lambda = %.2g" % (centre_proportions[best_h],scales[best_i],reg_params[best_j]))
#    # now we can plot the error for different scales using the best
#    # regulariation choice
#    
#    
#    print(scales)
#    print("____________________")
#    print(scales[best_i])
#    
#    
#    fig , ax = plot_train_test_errors(
#        "scale", scales, train_errors[:,best_j], test_errors[:,best_j])
#    ax.set_xscale('log')
#    # ...and the error for  different regularisation choices given the best
#    # scale choice 
#    fig , ax = plot_train_test_errors(
#        "$\lambda$", reg_params, train_errors[best_i,:], test_errors[best_i,:])
#    ax.set_xscale('log')
#    ax.set_ylim([0,20])

def import_data(ifname, delimiter=None, has_header=False, columns=None):
    """
    Imports a tab/comma/semi-colon/... separated data file as an array of 
    floating point numbers. If the import file has a header then this should
    be specified, and the field names will be returned as the second argument.

    parameters
    ----------
    ifname -- filename/path of data file.
    delimiter -- delimiter of data values
    has_header -- does the data-file have a header line
    columns -- a list of integers specifying which columns of the file to import
        (counting from 0)

    returns
    -------
    data_as_array -- the data as a numpy.array object  
    field_names -- if file has header, then this is a list of strings of the
      the field names imported. Otherwise, it is a None object.
    """
    if delimiter is None:
        delimiter = '\t'
    with open(ifname, 'r') as ifile:
        datareader = csv.reader(ifile, delimiter=delimiter)
        # if the data has a header line we want to avoid trying to import it.
        # instead we'll print it to screen
        if has_header:
            field_names = next(datareader)
            print("Importing data with field_names:\n\t" + ",".join(field_names))
        else:
            # if there is no header then the field names is a dummy variable
            field_names = None
        # create an empty list to store each row of data
        data = []
        for row in datareader:
            print("row = %r" % (row,))
            # for each row of data only take the columns we are interested in
            if not columns is None:
                row = [row[c] for c in columns]
            # now store in our data list
            data.append(row)
        print("There are %d entries" % len(data))
        print("Each row has %d elements" % len(data[0]))
    # convert the data (list object) into a numpy array.
    data_as_array = np.array(data).astype(float)
    if not columns is None and not field_names is None:
        # thin the associated field names if needed
        field_names = [field_names[c] for c in columns]
    # return this array to caller (and field_names if provided)
    return data_as_array, field_names


if __name__ == '__main__':
    """
    To run this script on just synthetic data use:

        python regression_external_data.py

    You can pass the data-file name as the first argument when you call
    your script from the command line. E.g. use:

        python regression_external_data.py datafile.tsv

    If you pass a second argument it will be taken as the delimiter, e.g.
    for comma separated values:

        python regression_external_data.py comma_separated_data.csv ","

    for semi-colon separated values:

        python regression_external_data.py comma_separated_data.csv ";"

    If your data has more than 2 columns you must specify which columns
    you wish to plot as a comma separated pair of values, e.g.

        python regression_external_data.py comma_separated_data.csv ";" 8,9

    For the wine quality data you will need to specify which columns to pass.
    """
    import sys
    if len(sys.argv) == 1:
        main() # calls the main function with no arguments
    elif len(sys.argv) == 2:
        # assumes that the first argument is the input filename/path
        main(ifname=sys.argv[1])
    elif len(sys.argv) == 3:
        # assumes that the second argument is the data delimiter
        main(ifname=sys.argv[1], delimiter=sys.argv[2])
    elif len(sys.argv) == 4:
        # assumes that the third argument is the list of columns to import
        columns = list(map(int, sys.argv[3].split(","))) 
        main(ifname=sys.argv[1], delimiter=sys.argv[2], columns=columns)

