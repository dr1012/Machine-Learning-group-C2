import numpy as np
import csv
import matplotlib.pyplot as plt

from regression_models import construct_rbf_feature_mapping

from regression_train_test import train_and_test_split
from regression_train_test import train_and_test_partition
#from regression_train_test import train_and_test
from regression_train_test import simple_evaluation_linear_model
#from regression_train_test import cv_evaluation_linear_model
from regression_train_test import create_cv_folds

from regression_plot import exploratory_plots
from regression_plot import plot_train_test_errors


#______________________________________________________________________________

with open('winequality-red.csv', 'r') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        header = next(datareader)
        data = []
       
        for row in datareader:
            row_of_floats = list(map(float, row))
            data.append(row_of_floats)

        # data is  of type list
        data_as_array = np.array(data)
        
        
        
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
    
    print('data shape')
    N = data.shape[0]

    inputs = data[:,0:11]
    targets = data[:,-1]

    #trains the rbf regression and identifies the optimal parameters
    run_rbf_model()
    
    
    
    
    
    
def run_rbf_model():
    
    
    N = data_as_array.shape[0]    
    print("___________________________________")
    print(N)
    print("___________________________________")    
    inputs = data_as_array[:,0:11]
    targets = data_as_array[:,-1]
    
    test_fraction = .15
    normalise_data_bool = 0
    
    
    #runs the training once on the non-normalised and then normalised feature data
    for j in range(2):
        if(normalise_data_bool == 0):
            
            print('______________________________non-normalised inputs_________________________________')
            print(inputs)
            print('______________________________non-normalised inputs_________________________________')

           n_norm_scale = parameter_search_rbf(inputs, targets, test_fraction)
            normalise_data_bool = 1
            
        else:
            
            #normalise input values
            for i in range(inputs.shape[1]):
                inputs[:,i] = ((inputs[:,i] - np.mean(inputs[:,i]))/  np.std(inputs[:,i]))
                
            print('______________________________normalised inputs_________________________________')
            print(inputs)
            print('______________________________normalised inputs_________________________________')
                
                
            parameter_search_rbf(inputs, targets, test_fraction)
    
    
    #compare the normalised vs non-normalised models on the final test set
    
    
    
    
#    train_error_linear, test_error_linear = evaluate_linear_approx(inputs, targets, test_fraction)
#    evaluate_rbf_for_various_reg_params(inputs, targets, test_fraction, test_error_linear)
    
#    parameter_search_rbf(inputs, targets, test_fraction)


    plt.show()


def parameter_search_rbf(inputs, targets, test_fraction):
    """
    """
    N = inputs.shape[0]
    folds_num = 5
    center_nums = 5
    
    
    #selection of centre_proportions to be varied     
    sample_fractions = np.array([0.05,0.1,0.15,0.2])    

    #parameters to be optimised
    scales = np.logspace(0,4, 20) # of the basis functions
    print('scales: %r' % scales)
    
    reg_params = np.logspace(-15,-1, 11) # choices of regularisation strength
    print('reg_params: %r' % reg_params)
    
    print('fitting the rbf model...')
    
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
                    
    print("optimal test_error = %r" %min_test_error,"optimal scale = %r" %scales[optimal_i], 
          "optimal centres: %r" % sample_fractions[optimal_h],
          "optimal lambda = %r" %reg_params[optimal_j] )



    return scales[optimal_i], sample_fractions[optimal_h],reg_params[optimal_j]


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
    
    

def train_and_test(
        train_inputs, train_targets, test_inputs, test_targets, reg_param=None):
    """
    Will fit a linear model with either least squares, or regularised least 
    squares to the training data, then evaluate on both test and training data

    parameters
    ----------
    train_inputs - the input design matrix for training
    train_targets - the training targets as a vector
    test_inputs - the input design matrix for testing
    test_targets - the test targets as a vector
    reg_param (optional) - the regularisation strength. If provided, then
        regularised maximum likelihood fitting is uses with this regularisation
        strength. Otherwise, (non-regularised) least squares is used.

    returns
    -------
    train_error - the training error for the approximation
    test_error - the test error for the approximation
    weights - the coefficient / weights of the model
    """
    # Find the optimal weights (depends on regularisation)
    if reg_param is None:
        # use simple least squares approach
        weights = ml_weights(
            train_inputs, train_targets)
    else:
        # use regularised least squares approach
        weights = regularised_ml_weights(
          train_inputs, train_targets,  reg_param)
    # predictions are linear functions of the inputs, we evaluate those here
    train_predicts = linear_model_predict(train_inputs, weights)
    test_predicts = linear_model_predict(test_inputs, weights)
    # evaluate the error between the predictions and true targets on both sets
    train_error = root_mean_squared_error(train_targets, train_predicts)
    test_error = root_mean_squared_error(test_targets, test_predicts)
    if np.isnan(test_error):
        print("test_predicts = %r" % (test_predicts,))
    return train_error, test_error, weights
    


def cv_evaluation_linear_model(
        inputs, targets, folds, reg_param=None):
    """
    Will split inputs and targets into train and test parts, then fit a linear
    model to the training part, and test on the both parts.

    Inputs can be a data matrix (or design matrix), targets should
    be real valued.

    parameters
    ----------
    inputs - the input design matrix (any feature mapping should already be
        applied)
    targets - the targets as a vector
    num_folds - the number of folds
    reg_param (optional) - the regularisation strength. If provided, then
        regularised least squares fitting is uses with this regularisation
        strength. Otherwise, (non-regularised) least squares is used.

    returns
    -------
    train_errors - the training errors for the approximation
    test_errors - the test errors for the approximation
    """
    # get the number of datapoints
    N = inputs.shape[0]
    # get th number of folds
    num_folds = len(folds)
    train_errors = np.empty(num_folds)
    test_errors = np.empty(num_folds)
    weights = []
    
    for f,fold in enumerate(folds):
        # f is the fold id, fold is the train-test split
        train_part, test_part = fold
        # break the data into train and test sets
        train_inputs, train_targets, test_inputs, test_targets = \
            train_and_test_partition(inputs, targets, train_part, test_part)
        # now train and evaluate the error on both sets
        train_error, test_error, weight = train_and_test(
            train_inputs, train_targets, test_inputs, test_targets,
            reg_param=reg_param)
        #print("train_error = %r" % (train_error,))
        #print("test_error = %r" % (test_error,))
        train_errors[f] = train_error
        test_errors[f] = test_error
        weights.append(weight)
    return train_errors, test_errors, weights
    
    

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
#            print("row = %r" % (row,))
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
