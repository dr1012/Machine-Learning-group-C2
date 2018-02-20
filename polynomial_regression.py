import csv
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg

from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter




# for creating synthetic data
from regression_samples import arbitrary_function_1
from regression_samples import arbitrary_function_2
from regression_samples import sample_data
# for performing regression


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



def expand_to_monomials(inputs, degree):
  
    total_array = []
    a = (inputs[:][:,0:1])
    b = (inputs[:][:,1:2])
    c = (inputs[:][:,2:3])
    d = (inputs[:][:,3:4])
    e = (inputs[:][:,4:5])
    f = (inputs[:][:,5:6])
    g = (inputs[:][:,6:7])
    h = (inputs[:][:,7:8])
    l = (inputs[:][:,8:9])
    m = (inputs[:][:,9:10])
    n = (inputs[:][:,10:11])
    expanded_inputs = []
    for k in range(degree+1):
        for j in range (k+1):
            for i in range (k-j+1):
                if i+j == k:
                    expanded_inputs.append((a**i)*(b**j))
    #  this array of arrays has each element as a row of the design matrix               
        total_array.append(expanded_inputs)
    return np.array(expanded_inputs).transpose()



def ml_weights(inputmtx, targets):
   
    Phi = np.matrix(inputmtx)
    targets = np.matrix(targets).reshape((len(targets),1))
    weights = linalg.inv(Phi.transpose()*Phi)*Phi.transpose()*targets
    return np.array(weights).flatten()

def construct_polynomial_approx(degree, weights):
   
    def prediction_function(xs):
        expanded_xs = np.matrix(expand_to_monomials(xs, degree))
        ys = expanded_xs*np.matrix(weights).reshape((len(weights),1))
        return np.array(ys).flatten()
   
    return prediction_function    


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
 


    degree_sequence=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

    for degree in degree_sequence:

    
        processed_inputs = expand_to_monomials(training_data_as_array, degree)

        weights = ml_weights(processed_inputs, train_targets)

        approx_func = construct_polynomial_approx(degree, weights)

        zs = approx_func(training_data_as_array)

        zs = np.meshgrid(zs)

        xs = training_data_as_array[:][:,0:1]

        ys = training_data_as_array[:][:,1:2]

        X, Y = np.meshgrid(training_data_as_array[:][:,0:1], training_data_as_array[:][:,1:2])

        '''
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs, ys, zs)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
     

        plt.show()
        '''
        train_error = root_mean_squared_error(train_targets, approx_func(training_data_as_array))
        test_error = root_mean_squared_error(test_targets, approx_func(validation_data_as_array))

        train_errors.append(train_error)
        test_errors.append(test_error)


    plot_train_test_errors("degree", degree_sequence, train_errors, test_errors,"2-D poly fit " + header[0] + " vs " + header[1])
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
