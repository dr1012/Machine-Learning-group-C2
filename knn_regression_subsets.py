import numpy as np
import csv
import math
import operator
import matplotlib.pyplot as plt

###################################################################################################################################
#This code has been based on:
#Title:Tutorial To Implement k-Nearest Neighbors in Python From Scratch
#Author:Jason Brownlee
#Date:September 12, 2014
#Link:https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
###################################################################################################################################

def conf_int(error_array):

   n = len(error_array)

   sigma = np.std(error_array)
   ste = sigma/n**0.5

   conf_low = error_array - ste
   conf_high = error_array + ste

   return conf_low, conf_high


def multi_knn(subset,training_data, test_data, max_k):

    subset.append(11)

    trainingSubset = training_data[:][:,subset]
    trainingSubsetList = trainingSubset.tolist()


    validationSubset  = test_data[:][:,subset]
    ValidationSubsetList =  validationSubset.tolist()

    #instance 1 and 2 are two points in n-dimensional space. They represent two separate rows in the table with n  being the number of parameter columns.
    def euclideanDistance(instance1, instance2, length):
        distance = 0
        for x in range(length):
            #this will give the sum of the squares of the distance between the 2 points in all the dimensions. 
            distance += pow((instance1[x]-instance2[x]), 2)
            # this will give us the Euclidean distance in (length) dimensions. 
        return math.sqrt(distance)


    def getNeighbors(trainingSet, testInstance, k):
        distances =  []
        #testInstance is a list representing a single point (single row in table). 
        #length is the number of dimensions
        length = len(testInstance)-1
        #for each row in training set
        #find the eculidean distance over all dimensions for that point 
        #the distances list will contain the distances between the test poiint and all the training points.
        #Actually it will contain the traininSet point (a list of all parameters) and the distance
        #then we sort it from smallest to largest distance
        for x in range(len(trainingSet)):
            dist = euclideanDistance(testInstance, trainingSet[x], length)
            distances.append((trainingSet[x], dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
            # neighbors is a list that will contain the data-points [list of parameters] of the k nearest points
            neighbors.append(distances[x][0])
        return neighbors

    def getResponse(neighbors,k):
        #a dictionary containing the different claasses and the number of neighbors that belong to that class
        total_sum = 0
        for x in range(len(neighbors)):
            response = neighbors[x][-1]
            total_sum += response
        return total_sum/k    
    # In case of a  draw this selects one response, ideally we would select an ubiased random response.


    #  Simple function that evaluates accuracy of predictions
    def rmsError(testSet, predictions):
        errors = []
        for x in range(len(testSet)):
            error = (testSet[x][-1]-predictions[x])
            errors.append(error**2) 
        rms_error =  math.sqrt(np.mean(errors))
        return rms_error

    def mean_absolute_percentage_error(testSet, predictions):
        errors = []
        for x in range(len(testSet)):
            individual__percentage_error = ((abs((testSet[x][-1]-predictions[x])))/testSet[x][-1])*100
            errors.append(individual__percentage_error)
        return np.mean(errors)    

            

    max_neighbours = max_k
    accuracies = []
    test_errors = []

    # this code is for plotting rms errors

    for y in range(1,max_neighbours):

        predictions = []
        for x in range(len(ValidationSubsetList)):
            neighbors = getNeighbors(training_data, ValidationSubsetList[x], y)
            result = getResponse(neighbors,y)
            predictions.append(result)

            
        accuracy = rmsError(ValidationSubsetList, predictions)

        test_errors.append(accuracy)
        accuracies.append([y,accuracy])

    npAcurracies = np.array(accuracies)

    del subset[-1]

    conf_low, conf_high = conf_int(test_errors)
    min_error = np.min(npAcurracies[:,1:2])
    min_index = np.argmin(npAcurracies[:,1:2])
    k_min_error = npAcurracies[min_index][0]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(npAcurracies[:,0:1],npAcurracies[:,1:2])
    ax.set_xlabel("K")
    ax.set_ylabel("$E_{RMS}$")
    ax.set_ylim(None,1)
    #ax.fill_between(npAcurracies[:,0:1], conf_low, conf_high, alpha=0.2, color='r')
    ax.set_title("KNN regression, subset: "+ str(subset) + " ,minimum error: " + str(min_error) + " at K = " + str(k_min_error),fontsize = 8)
    fig.savefig("KNN regression " + str(subset) +".pdf", fmt="pdf")

    return [subset, min_error, k_min_error]	





