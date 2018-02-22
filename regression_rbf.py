import csv
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg

from regression_samples import arbitrary_function_1

# for performing regression
from regression_models import construct_rbf_feature_mapping
from regression_models import construct_feature_mapping_approx
# for plotting results
from regression_plot import plot_function_data_and_approximation
from regression_plot import plot_train_test_errors
# for evaluating fit
from regression_train_test import train_and_test
from regression_train_test import train_and_test_split
from regression_train_test import train_and_test_partition

from regression_train_test import create_cv_folds
from regression_train_test import cv_evaluation_linear_model

with open('winequality-red-commas.csv', 'r') as csvfile:
    datareader = csv.reader(csvfile, delimiter=',')
    header = next(datareader)
    data = []

    for row in datareader:
        row_of_floats = list(map(float, row))
        data.append(row_of_floats)

    # data is  of type list
    data_as_array = np.array(data)


def evaluate_reg_param(inputs, targets, folds, centres, scale, reg_params=None):
    """
      Evaluate then plot the performance of different regularisation parameters
    """
    # create the feature mappoing and then the design matrix
    feature_mapping = construct_rbf_feature_mapping(centres,scale)
    designmtx = feature_mapping(inputs)
    # choose a range of regularisation parameters
    if reg_params is None:
        reg_params = np.logspace(-9,-4, 5)
    num_values = reg_params.size
    num_folds = len(folds)
    # create some arrays to store results
    train_mean_errors = np.zeros(num_values)
    test_mean_errors = np.zeros(num_values)
    train_stdev_errors = np.zeros(num_values)
    test_stdev_errors = np.zeros(num_values)
    #
    for r, reg_param in enumerate(reg_params):
        # r is the index of reg_param, reg_param is the regularisation parameter
        # cross validate with this regularisation parameter
        train_errors, test_errors = cv_evaluation_linear_model(
            designmtx, targets, folds, reg_param=reg_param)
        # we're interested in the average (mean) training and testing errors
        train_mean_error = np.mean(train_errors)
        test_mean_error = np.mean(test_errors)
        train_stdev_error = np.std(train_errors)
        test_stdev_error = np.std(test_errors)
        # store the results
        train_mean_errors[r] = train_mean_error
        test_mean_errors[r] = test_mean_error
        train_stdev_errors[r] = train_stdev_error
        test_stdev_errors[r] = test_stdev_error

    best_r = np.argmin(test_mean_errors)
        # np.argmin(test_mean_errors[r])

    print("Best Choice of Regularization Parameter from Cross-Validation:")
    print(
            "\tlambda %.2g" % (reg_params[best_r]))

    # Now plot the results
    fig, ax = plot_train_test_errors(
        "$\lambda$", reg_params, train_mean_errors, test_mean_errors)
    # Here we plot the error ranges too: mean plus/minus 1 standard error.
    # 1 standard error is the standard deviation divided by sqrt(n) where
    # n is the number of samples.
    # (There are other choices for error bars.)
    # train error bars
    lower = train_mean_errors - train_stdev_errors/np.sqrt(num_folds)
    upper = train_mean_errors + train_stdev_errors/np.sqrt(num_folds)
    ax.fill_between(reg_params, lower, upper, alpha=0.2, color='b')
    # test error bars
    lower = test_mean_errors - test_stdev_errors/np.sqrt(num_folds)
    upper = test_mean_errors + test_stdev_errors/np.sqrt(num_folds)
    ax.fill_between(reg_params, lower, upper, alpha=0.2, color='r')
    ax.set_xscale('log')


def evaluate_scale(inputs, targets, folds, centres, reg_param, scales=None):
    """
    evaluate then plot the performance of different basis function scales
    """
    # choose a range of scales
    if scales is None:
        scales = np.logspace(2,4,10)
    #
    num_values = scales.size
    num_folds = len(folds)
    # create some arrays to store results
    train_mean_errors = np.zeros(num_values)
    test_mean_errors = np.zeros(num_values)
    train_stdev_errors = np.zeros(num_values)
    test_stdev_errors = np.zeros(num_values)
    #
    for s, scale in enumerate(scales):
        feature_mapping = construct_rbf_feature_mapping(centres,scale)
        designmtx = feature_mapping(inputs)
        # r is the index of reg_param, reg_param is the regularisation parameter
        # cross validate with this regularisation parameter
        train_errors, test_errors = cv_evaluation_linear_model(
            designmtx, targets, folds, reg_param=reg_param)
        # we're interested in the average (mean) training and testing errors
        train_mean_error = np.mean(train_errors)
        test_mean_error = np.mean(test_errors)
        train_stdev_error = np.std(train_errors)
        test_stdev_error = np.std(test_errors)
        # store the results
        train_mean_errors[s] = train_mean_error
        test_mean_errors[s] = test_mean_error
        train_stdev_errors[s] = train_stdev_error
        test_stdev_errors[s] = test_stdev_error

    best_s = np.argmin(test_mean_errors)

    print("Best Choice of Scale Parameter from Cross-Validation:")
    print(
            "\tscale %.2g" % (scales[best_s]))

    # Now plot the results
    fig, ax = plot_train_test_errors(
        "scale", scales, train_mean_errors, test_mean_errors)
    # Here we plot the error ranges too: mean plus/minus 1 standard error.
    # 1 standard error is the standard deviation divided by sqrt(n) where
    # n is the number of samples.
    # (There are other choices for error bars.)
    # train error bars
    lower = train_mean_errors - train_stdev_errors/np.sqrt(num_folds)
    upper = train_mean_errors + train_stdev_errors/np.sqrt(num_folds)
    ax.fill_between(scales, lower, upper, alpha=0.2, color='b')
    # test error bars
    lower = test_mean_errors - test_stdev_errors/np.sqrt(num_folds)
    upper = test_mean_errors + test_stdev_errors/np.sqrt(num_folds)
    ax.fill_between(scales, lower, upper, alpha=0.2, color='r')
    ax.set_xscale('log')


def evaluate_num_centres(
        inputs, targets, folds, scale, reg_param, num_centres_sequence=None):
    """
      Evaluate then plot the performance of different numbers of basis
      function centres.
    """
    # fix the reg_param
    # reg_param = 0.08
    # fix the scale
    # scale = 0.03
    # choose a range of numbers of centres
    if num_centres_sequence is None:
        num_centres_sequence = np.arange(5,100)
    num_values = num_centres_sequence.size
    num_folds = len(folds)
    #
    # create some arrays to store results
    train_mean_errors = np.zeros(num_values)
    test_mean_errors = np.zeros(num_values)
    train_stdev_errors = np.zeros(num_values)
    test_stdev_errors = np.zeros(num_values)
    #
    # run the experiments
    for c, num_centres in enumerate(num_centres_sequence):
        centres = np.linspace(0,1,num_centres)
        feature_mapping = construct_rbf_feature_mapping(centres,scale)
        designmtx = feature_mapping(inputs)
        # r is the index of reg_param, reg_param is the regularisation parameter
        # cross validate with this regularisation parameter
        train_errors, test_errors = cv_evaluation_linear_model(
            designmtx, targets, folds, reg_param=reg_param)
        # we're interested in the average (mean) training and testing errors
        train_mean_error = np.mean(train_errors)
        test_mean_error = np.mean(test_errors)
        train_stdev_error = np.std(train_errors)
        test_stdev_error = np.std(test_errors)
        # store the results
        train_mean_errors[c] = train_mean_error
        test_mean_errors[c] = test_mean_error
        train_stdev_errors[c] = train_stdev_error
        test_stdev_errors[c] = test_stdev_error
    #
    # Now plot the results
    fig, ax = plot_train_test_errors(
        "Num. Centres", num_centres_sequence, train_mean_errors, test_mean_errors)
    # Here we plot the error ranges too: mean plus/minus 1 standard error.
    # 1 standard error is the standard deviation divided by sqrt(n) where
    # n is the number of samples.
    # (There are other choices for error bars.)
    # train error bars
    lower = train_mean_errors - train_stdev_errors/np.sqrt(num_folds)
    upper = train_mean_errors + train_stdev_errors/np.sqrt(num_folds)
    ax.fill_between(num_centres_sequence, lower, upper, alpha=0.2, color='b')
    # test error bars
    lower = test_mean_errors - test_stdev_errors/np.sqrt(num_folds)
    upper = test_mean_errors + test_stdev_errors/np.sqrt(num_folds)
    ax.fill_between(num_centres_sequence, lower, upper, alpha=0.2, color='r')

def parameter_search_rbf(inputs, targets, test_fraction):
    """
    """
    N = inputs.shape[0]
    # run all experiments on the same train-test split of the data
    train_part, test_part = train_and_test_split(N, test_fraction=test_fraction)
    # for the centres of the basis functions sample 10% of the data
    sample_fraction = 0.15
    p = (1-sample_fraction,sample_fraction)
    centres = inputs[np.random.choice([False,True], size=N, p=p),:]
    # print(centres)
    print("centres.shape = %r" % (centres.shape,))
    scales = np.logspace(2,4, 10) # of the basis functions
    reg_params = np.logspace(-9,-4, 5) # choices of regularisation strength
    # create empty 2d arrays to store the train and test errors
    train_errors = np.empty((scales.size,reg_params.size))
    test_errors = np.empty((scales.size,reg_params.size))
    # iterate over the scales
    for i,scale in enumerate(scales):
        # i is the index, scale is the corresponding scale
        # we must recreate the feature mapping each time for different scales
        feature_mapping = construct_rbf_feature_mapping(centres,scale)
        designmtx = feature_mapping(inputs)
        # partition the design matrix and targets into train and test
        train_designmtx, train_targets, test_designmtx, test_targets = \
            train_and_test_partition(
                designmtx, targets, train_part, test_part)
        # iteratre over the regularisation parameters
        for j, reg_param in enumerate(reg_params):
            # j is the index, reg_param is the corresponding regularisation
            # parameter
            # train and test the data
            train_error, test_error = train_and_test(
                train_designmtx, train_targets, test_designmtx, test_targets,
                reg_param=reg_param)
            # store the train and test errors in our 2d arrays
            train_errors[i,j] = train_error
            test_errors[i,j] = test_error
    # we have a 2d array of train and test errors, we want to know the (i,j)
    # index of the best value
    best_i = np.argmin(np.argmin(test_errors,axis=1))
    best_j = np.argmin(test_errors[i,:])
    print("Best joint choice of parameters:")
    print(
        "\tscale %.2g and lambda = %.2g" % (scales[best_i],reg_params[best_j]))
    # now we can plot the error for different scales using the best
    # regulariation choice
    fig , ax = plot_train_test_errors(
        "scale", scales, train_errors[:,best_j], test_errors[:,best_j])
    ax.set_xscale('log')
    # ...and the error for  different regularisation choices given the best
    # scale choice
    fig , ax = plot_train_test_errors(
        "$\lambda$", reg_params, train_errors[best_i,:], test_errors[best_i,:])
    ax.set_xscale('log')
#    ax.set_ylim([0,20])

    return scales[best_i],reg_params[best_j]

def main():
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

    #retrieve train targets and inputs from data array
    N = data_as_array.shape[0]
    targets = data_as_array[:,-1]
    inputs = data_as_array[:,[0,1,2,3,4,5,6,7,8,9,10]]

    # normalise inputs (meaning radial basis functions are more helpful)
    inputs[:, 0] = (inputs[:, 0] - np.mean(inputs[:, 0])) / np.std(inputs[:, 0])
    inputs[:, 1] = (inputs[:, 1] - np.mean(inputs[:, 1])) / np.std(inputs[:, 1])
    inputs[:, 2] = (inputs[:, 2] - np.mean(inputs[:, 2])) / np.std(inputs[:, 2])
    inputs[:, 3] = (inputs[:, 3] - np.mean(inputs[:, 3])) / np.std(inputs[:, 3])
    inputs[:, 4] = (inputs[:, 4] - np.mean(inputs[:, 4])) / np.std(inputs[:, 4])
    inputs[:, 5] = (inputs[:, 5] - np.mean(inputs[:, 5])) / np.std(inputs[:, 5])
    inputs[:, 6] = (inputs[:, 6] - np.mean(inputs[:, 6])) / np.std(inputs[:, 6])
    inputs[:, 7] = (inputs[:, 7] - np.mean(inputs[:, 7])) / np.std(inputs[:, 7])
    inputs[:, 8] = (inputs[:, 8] - np.mean(inputs[:, 8])) / np.std(inputs[:, 8])
    inputs[:, 9] = (inputs[:, 9] - np.mean(inputs[:, 9])) / np.std(inputs[:, 9])
    # inputs[:, 10] = (inputs[:, 10] - np.mean(inputs[:, 10])) / np.std(inputs[:, 10])

    test_fraction = 0.25

    # specify the centres of the rbf basis functions
    centres = inputs[np.random.choice([False, True], size=N, p=[0.9, 0.1]), :]

    # the width (analogous to standard deviation) of the basis functions
    scale = 0.15

    best_scale, best_param = parameter_search_rbf(inputs, targets, test_fraction)

    # get the cross-validation folds
    num_folds = 5
    folds = create_cv_folds(N, num_folds)

    evaluate_scale(inputs, targets, folds, centres, best_param)
    evaluate_reg_param(inputs, targets, folds, centres, best_scale)
    # evaluate_num_centres(inputs, targets, folds, best_scale, best_param)

    plt.show()

if __name__ == '__main__':
    # this bit only runs when this script is called from the command line
    main()
