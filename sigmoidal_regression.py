import numpy as np
import csv
import math
import operator
import matplotlib.pyplot as plt
import numpy.linalg as linalg


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




centres = []
scales = []
for x in range(0,11):
    centres.append(np.mean(training_data_as_array[:,x:x+1]))
    scales.append(np.std(training_data_as_array[:,x:x+1]))

centres = np.array(centres)
scales = np.array(scales)

def construct_sigmoidal_feature_mapping(centres, scales):
    """
    parameters
    ----------
    centres - a DxM matrix (numpy array) where D is the dimension of the space
        and each row is the central position of an rbf basis function.
        For D=1 can pass an M-vector (numpy array).
    scale - a float determining the width of the distribution. Equivalent role
        to the standard deviation in the Gaussian distribution.

    returns
    -------
    feature_mapping - a function which takes an NxD data matrix and returns
        the design matrix (NxM matrix of features)
    """
    #  to enable python's broadcasting capability we need the centres
    # array as a 1xDxM array
    if len(centres.shape) == 1:
        centres = centres.reshape((1,1,centres.size))
    else:
        centres = centres.reshape((1,centres.shape[1],centres.shape[0]))
    # the denominator

    if len(scales.shape) == 1:
        scales = scales.reshape((1,1,scales.size))
    else:
        scales = scales.reshape((1,scales.shape[1],scales.shape[0]))



    # now create a function based on these basis functions
    def feature_mapping(datamtx):
        #  to enable python's broadcasting capability we need the datamtx array
        # as a NxDx1 array
        if len(datamtx.shape) == 1:
            # if the datamtx is just an array of scalars, turn this into
            # a Nx1x1 array
            datamtx = datamtx.reshape((datamtx.size,1,1))
        else:
            # if datamtx is NxD array, then we reshape matrix as a
            # NxDx1 array
            datamtx = datamtx.reshape((datamtx.shape[0], datamtx.shape[1], 1))
            a = (datamtx-centres)/(scales)

        return 1/(1+np.exp(-a))
    # return the created function
    return feature_mapping

def ml_weights(inputmtx, targets):
    """
    This method returns the weights that give the best linear fit between
    the processed inputs and the targets.
    """
    Phi = np.matrix(inputmtx)
    print("Phi")
    print(Phi)
    targets = np.matrix(targets).reshape((len(targets),1))
    print("targets")
    print(targets)
    weights = linalg.inv(Phi.transpose()*Phi)*Phi.transpose()*targets
    return np.array(weights).flatten()

def construct_feature_mapping_approx(feature_mapping, weights):
    """
    This function creates and returns a prediction function based on a
    feature mapping and some weights.

    The returned prediction function takes a set of input values and returns
    the predicted output for each.
    """
    # here is a function that is created on the fly from the input feature
    # mapping and weights
    def prediction_function(xs):
        designmtx = np.matrix(feature_mapping(xs))
        return linear_model_predict(designmtx, weights)
    # we return the function reference (handle) itself. This can be used like
    # any other function
    return prediction_function    

def linear_model_predict(designmtx, weights):
    ys = np.matrix(designmtx)*np.matrix(weights).reshape((len(weights),1))
    return np.array(ys).flatten()

def main():
    targets =  training_data_as_array[:][:,11:12]
    feature_mapping = construct_sigmoidal_feature_mapping(centres,scales)
    datamtx = training_data_as_array[:][:,0:1]
    designmtx = feature_mapping(datamtx)
    

    weights =  ml_weights(designmtx,targets)
'''   
    sigmoidal_approx = construct_feature_mapping_approx(feature_mapping, weights)

    print(sigmoidal_approx)
'''   



    

if __name__ == '__main__':
    # this bit only runs when this script is called from the command line
    main()
        
