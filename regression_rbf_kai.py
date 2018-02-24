import numpy as np
import csv
import matplotlib.pyplot as plt

from regression_models import ml_weights
from regression_models import regularised_ml_weights
from regression_models import linear_model_predict
from regression_models import construct_rbf_feature_mapping
from regression_models import construct_feature_mapping_approx

from regression_train_test import root_mean_squared_error
from regression_train_test import train_and_test_split
from regression_train_test import train_and_test_partition
#from regression_train_test import train_and_test
from regression_train_test import simple_evaluation_linear_model
#from regression_train_test import cv_evaluation_linear_model
from regression_train_test import create_cv_folds

from regression_plot import exploratory_plots
from regression_plot import plot_train_test_errors


#______________________________________________________________________________

with open('final_training_data.csv', 'r') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        header = next(datareader)
        data = []
       
        for row in datareader:
            row_of_floats = list(map(float, row))
            data.append(row_of_floats)

        # data is  of type list
        data_as_array = np.array(data)
        
        
with open('final_test_data.csv', 'r') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        header = next(datareader)
        data = []
       
        for row in datareader:
            row_of_floats = list(map(float, row))
            data.append(row_of_floats)

        # data is  of type list
        test_data_as_array = np.array(data)
        
        
        
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
    
    #trains the rbf regression and identifies the optimal parameters
    run_rbf_model()
    
    
def run_rbf_model():
    """
    To be called when the script is run. This function trains and evaluates the 
    linear regression model with Radial Basis Functions.
    """
    
    
    #ensures re-producability of our results
    np.random.seed(5)
    
    #training data
    inputs = data_as_array[:,0:11]
    targets = data_as_array[:,-1]
    
    #testing data
    test_inputs = test_data_as_array[:,0:11]
    test_targets = test_data_as_array[:,-1]

    #selection of centre_proportions to be varied     
    sample_fractions = np.array([0.05,0.1,0.15,0.2])    


    normalise_data_bool = 0
    normalised_min_prediction = 0
    non_normalised_min_prediction = 0
    
    #runs the training once on the non-normalised and then normalised feature data
    for j in range(2):

        #training the model with non-normalised data                            
        if(normalise_data_bool == 0):       
            #training the model and returning optimal 2nd order parameters - non-normalised data

            #old returned values
#            scale, centers, reg_param, optimal_weights, optimal_feature_mapping, train_errors, test_errors, optimal_h, optimal_i, optimal_j = parameter_search_rbf(inputs, targets, sample_fractions)    
            scales, centers, reg_params, optimal_weights, optimal_feature_mapping, train_errors, test_errors, optimal_h, optimal_i, optimal_j = parameter_search_rbf(inputs, targets, sample_fractions)    
            
            
            
            #testing the model's performance
            predict_func = construct_feature_mapping_approx(optimal_feature_mapping, optimal_weights)
            non_normalised_min_prediction = root_mean_squared_error(test_targets , predict_func(test_inputs))
            print("non-normalised-data: final model testing prediction error: ")
            print(non_normalised_min_prediction)

            normalise_data_bool = 1            

        #training the model with normalised data
        elif(normalise_data_bool == 1):
            #normalise the input data
            for i in range(inputs.shape[1]):
                inputs[:,i] = ((inputs[:,i] - np.mean(inputs[:,i]))/  np.std(inputs[:,i]))
            
            #training the model and returning optimal 2nd order parameters - normalised data
            n_scales, n_centers, n_reg_params, n_optimal_weights, n_optimal_feature_mapping, n_train_errors, n_test_errors, n_optimal_h, n_optimal_i, n_optimal_j = parameter_search_rbf(inputs, targets, sample_fractions)    

            #testing the model's performance
            predict_func = construct_feature_mapping_approx(n_optimal_feature_mapping, n_optimal_weights)
            for i in range(inputs.shape[1]):
                test_inputs[:,i] = ((test_inputs[:,i] - np.mean(test_inputs[:,i]))/  np.std(test_inputs[:,i]))
            
            normalised_min_prediction = root_mean_squared_error(test_targets , predict_func(test_inputs))
            print("normalised-data: final model testing prediction error: ")
            print(normalised_min_prediction)
                        
        else:
            print("error in evaluating the RBF model")
        
    
    
    #plot train and test error vs sample fractions
    fig , ax = plot_train_test_errors_kai("sample fractions", sample_fractions, train_errors[:,optimal_i,optimal_j], test_errors[:,optimal_i,optimal_j], 
                                          sample_fractions, n_train_errors[:,n_optimal_i,n_optimal_j], n_test_errors[:,n_optimal_i,n_optimal_j])
    
    plt.title('Parameter optimisation - the behaviour of $E_{RMS}$ for sample fractions')
    plt.savefig("RBF optimisation - number of centers.pdf", bbox_inches='tight')
    ax.set_xlim([0,0.25])        
        
    #plot train and test error vs scale   
    fig , ax = plot_train_test_errors_kai("scale", scales, train_errors[optimal_h,:,optimal_j], test_errors[optimal_h,:,optimal_j],
                                          n_scales,n_train_errors[n_optimal_h,:,n_optimal_j], test_errors[n_optimal_h,:,n_optimal_j] )
    ax.set_xscale('log')
    ax.set_ylim([0,1.5])
    plt.title('Parameter optimisation - the behaviour of $E_{RMS}$ for changing scales')
    plt.savefig("RBF optimisation - scales.pdf", bbox_inches='tight')

    #plot train and test error vs lambda
    fig , ax = plot_train_test_errors_kai("$\lambda$", reg_params, train_errors[optimal_h,optimal_i,:], test_errors[optimal_h,optimal_i,:],
                                      n_reg_params, n_train_errors[n_optimal_h,n_optimal_i,:], n_test_errors[n_optimal_h,n_optimal_i,:])
    ax.set_xscale('log')
    ax.set_ylim([0,1.5])
    plt.title('Parameter optimisation - the behaviour of $E_{RMS}$ for changing $\lambda$')
    plt.savefig("RBF optimisation - $\lambda$.pdf", bbox_inches='tight')

        
#    plt.show()


def parameter_search_rbf(inputs, targets, sample_fractions):
    """
    """
    N = inputs.shape[0]
    print(N)
    folds_num = 5
    center_nums = 5
    

    #parameters to be optimised
    scales = np.logspace(0,4, 20) # of the basis functions
    reg_params = np.logspace(-15,-1, 11) # choices of regularisation strength
    
    print('fitting the rbf model...')
    
    #create folds to run cross-validation on the parameters to be optimised
    folds = create_cv_folds(N,folds_num)
 
    # create empty 3d arrays to store the train and test errors
    train_errors = np.empty((sample_fractions.size, scales.size,reg_params.size))
    test_errors = np.empty((sample_fractions.size, scales.size,reg_params.size))

    #create container variables to store the optimal solution from the loops
    optimised_ml_weights = np.empty(inputs[0].shape)
    optimal_feature_mapping = 0
    #store the location of the optimal error value in the error matrix
    optimal_h = 0    
    optimal_i = 0
    optimal_j = 0
    
    min_test_error = 10*10
    
    #tripple for loop iterates over the number of centers, scales and regularization
    #parameter, re-trains the model for every instance in the loop, cross-validates the error
    #and returns the 'optimal' error of each iteration
    
    #iterate over different number of centres
    for h, sample_fraction in enumerate(sample_fractions):
        # h is the index, sample_fraction is gives the number of centers to be chosen
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
                    optimised_ml_weights = weight
                    optimal_feature_mapping = feature_mapping
                    

    print("crossvalidation optimal test_error = %r" %min_test_error,"optimal scale = %r" %scales[optimal_i], 
          "optimal centres: %r" % sample_fractions[optimal_h],
          "optimal lambda = %r" %reg_params[optimal_j])


    return scales, sample_fractions,reg_params ,optimised_ml_weights, optimal_feature_mapping, train_errors, test_errors, optimal_h, optimal_i, optimal_j



#TODO: REWRITE SO THAT IT CAN PLOT ALL FOUR LINES (NORMALISED AND NON-NORMALISED IN ONE GRAPH)
def plot_train_test_errors_kai(
        control_var, experiment_sequence, train_errors, test_errors, n_experiment_sequence, n_train_errors, n_test_errors):
    """
    Plot the train and test errors for a sequence of experiments.

    parameters
    ----------
    control_var - the name of the control variable, e.g. degree (for polynomial)
        degree.
    experiment_sequence - a list of values applied to the control variable.
    """
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    train_line, = ax.plot(experiment_sequence, train_errors,'b-', label='non-normalised train')
    test_line, = ax.plot(experiment_sequence, test_errors, 'r-', label='non-normalised test')
    n_train_line, = ax.plot(n_experiment_sequence, n_train_errors,'c--', label='normalised train')
    n_test_line, = ax.plot(n_experiment_sequence, n_test_errors, 'm--', label='normalised test')

    plt.axhline(y=0.65, color='black', linestyle='dashdot', label='baseline')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    ax.set_xlabel(control_var)
    ax.set_ylabel("$E_{RMS}$")

    return fig, ax


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
        weights = weight 
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
