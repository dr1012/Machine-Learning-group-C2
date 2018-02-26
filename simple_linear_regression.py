
import csv
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg


def simple_linear_main():


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

    train_inputs = internal_training_data_as_nparray[:][:,0:11]
    train_targets = internal_training_data_as_nparray[:][:,11:12]

    test_inputs = internal_test_data_as_nparray[:][:,0:11]
    test_targets = internal_test_data_as_nparray[:][:,11:12]









    weights = ml_weights(train_inputs,train_targets)

    train_predicts = linear_model_predict(train_inputs, weights)
    test_predicts = linear_model_predict(test_inputs, weights)




    train_error = root_mean_squared_error(train_targets, train_predicts)
    test_error = root_mean_squared_error(test_targets, test_predicts)

    print("non-regularised simple linear regression training error: " + str(train_error))

    print("non-regularised simple linear regression test error: " +  str(test_error))

    return train_error, test_error, weights

def ml_weights(inputmtx, targets):

    Phi = np.matrix(inputmtx)
    targets = np.matrix(targets).reshape((len(targets),1))
    weights = linalg.inv(Phi.transpose()*Phi)*Phi.transpose()*targets
    return np.array(weights).flatten()




def linear_model_predict(designmtx, weights):
    ys = np.matrix(designmtx)*np.matrix(weights).reshape((len(weights),1))
    return np.array(ys).flatten()

  

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
