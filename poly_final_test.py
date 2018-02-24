import csv
import numpy as np
import matplotlib.pyplot as plt


from regression_models import expand_to_monomials
from regression_models import ml_weights
from regression_models import construct_polynomial_approx
# for plotting results
from regression_plot import plot_function_data_and_approximation





def final_poly_test_function(polynomial_function, subset, degree, test_data):

    degree_sequence = range(0,int(degree+2))
    
    print("function type" + str(type(polynomial_function)))
   

    test_degree_error_pairs = []
    test_errors = []

    test_inputs = test_data[:][:,subset]
    test_targets = test_data[:][:,11:12]

    approx_func  = polynomial_function

    for degree in degree_sequence:

        test_error = root_mean_squared_error(test_targets, approx_func(test_inputs))

        test_errors.append(test_error)

        test_degree_error_pairs.append([degree, test_error])


    
    npTestErrors  = np.array(test_degree_error_pairs)
    min_error = np.min(npTestErrors[:,1:2])
    min_index = np.argmin(npTestErrors[:,1:2])
    degree_min_error = npTestErrors[min_index][0]

    

    title =  "FINAL poly fit of " + str(subset) + " min rms error: " + str(min_error) + " at degree = " + str(degree_min_error)
    plot_train_test_errors("degree", degree_sequence, test_errors, title)
    plt.savefig("FINAL " +str(len(subset))+"-D_poly_"+str(subset)+".pdf", fmt="pdf")

    return min_error, degree_min_error








    

 









def plot_train_test_errors(
        control_var, experiment_sequence, test_errors, title):
    """
    inputs
    ------
    control_var - the name of the control variable, e.g. degree (for polynomial)
        degree.
    experiment_sequence - a list of values applied to the control variable.
    """
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    test_line, = ax.plot(experiment_sequence, test_errors, 'r-')
    ax.set_xlabel(control_var)
    ax.set_ylabel("$E_{RMS}$")
    ax.set_title(title, fontsize = 5)
    ax.legend([test_line], ["test"])
    # errors won't strictly lie below 1 but very large errors can make the plot
    # difficult to read. So we restrict the limits of the y-axis.
    ax.set_ylim((0,1))
    return fig, ax




  

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


