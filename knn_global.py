import numpy as np
import csv
import math
import operator
import matplotlib.pyplot as plt
import itertools

from knn_regression_subsets import multi_knn

###################################################################################################################################
#This code has been based on:
#Title:Tutorial To Implement k-Nearest Neighbors in Python From Scratch
#Author:Jason Brownlee
#Date:September 12, 2014
#Link:https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

#and on:

#This code has been based on:
#Title:How to get all possible combinations of a list’s elements?
#Author:Dan H
#Date:May 5, 2011
#Link:https://stackoverflow.com/questions/464864/how-to-get-all-possible-combinations-of-a-list-s-elements



###################################################################################################################################




def main_knn_function():
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





    def conf_int(error_array):

        n = len(error_array)

        sigma = np.std(error_array)
        ste = sigma/n**0.5

        conf_low = error_array - ste
        conf_high = error_array + ste
        return conf_low, conf_high


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
           
            neighbors.append(distances[x][0])
        return neighbors


    def getResponse(neighbors,k):
      
        total_sum = 0
        for x in range(len(neighbors)):
            response = neighbors[x][-1]
            total_sum += response
        return total_sum/k    


    def rmsError(testSet, predictions):
        errors = []
        for x in range(len(testSet)):
            error = (testSet[x][-1]-predictions[x])
            errors.append(error**2) 
        rms_error =  math.sqrt(np.mean(errors))
        return rms_error
    

    #########getting the inidividual atribute KNN models##########################
    total_minErrors_and_optimmal_k = []

    for a in range(0,11):

            trainingSubset = internal_training_data_as_nparray[:][:,[a,11]]
            trainingSubsetList = trainingSubset.tolist()


            validationSubset  = internal_test_data_as_nparray[:][:,[a,11]]
            ValidationSubsetList =  validationSubset.tolist()

            max_neighbours = 80
            accuracies = []
            test_errors = []
            

            for y in range(1,max_neighbours):

                predictions = []
                for x in range(len(internal_test_data_as_nparray)):
                    neighbors = getNeighbors(trainingSubsetList, ValidationSubsetList[x], y)
                    result = getResponse(neighbors,y)
                    predictions.append(result)
                
                    
                accuracy = rmsError(internal_test_data_as_nparray, predictions)
                
                test_errors.append(accuracy)
                accuracies.append([y,accuracy])

            npAcurracies = np.array(accuracies)

            
            min_error = np.min(npAcurracies[:,1:2])
            min_index = np.argmin(npAcurracies[:,1:2])
            K_min_error = npAcurracies[min_index][0]

            total_minErrors_and_optimmal_k.append([float(a), min_error, float(K_min_error)])

            conf_low, conf_high = conf_int(test_errors)
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.plot(npAcurracies[:,0:1],npAcurracies[:,1:2])
            ax.set_ylim(None,1)
            ax.set_xlabel("K")
            ax.set_ylabel("$E_{RMS}$")
            ax.set_title("KNN for " + header[a] +" ,minimum error: " + str(min_error) + " at K = " + str(K_min_error)  ,fontsize = 8)
           # ax.fill_between(list(npAcurracies[:,0:1]), conf_low, conf_high, alpha=0.2, color='r')
            fig.savefig("KNN_individual_"+header[a]+".pdf", fmt="pdf")
            

    ################find the best 4 attributes and generating all possible subets####################


    #array contains [attribute number, minimum error, k for minimum error]

    np_individual_errors_array =  np.array(total_minErrors_and_optimmal_k)
    # print(np_individual_errors_array)


    #sorted array by which attribute has the smallest error
    sorted_array = np_individual_errors_array[np_individual_errors_array[:,1].argsort()]


    # taking the  best 4 attributes as a subset
    subset_array = sorted_array[0:3]

    subset_attributes =  (subset_array[:][:,0:1]).flatten()


    ############testing all combinations of the subset of the best attributes ###################################


    subset_error_k = []
    for L in range(0, len(subset_attributes)+1):
        for subset in itertools.combinations(subset_attributes, L):
    #every subset here is a tuple containing the indexes of the attributes in the subset
            if(len(subset)>1):
                # print(subset)
                subset_list_temp = list(subset)
                subset_list =[int(i) for i in subset_list_temp]
                subset_error_k.append(multi_knn(subset_list,internal_training_data_as_nparray,internal_test_data_as_nparray,80))

    final_array = np.array(subset_error_k)

    # print(final_array)

    sorted_final_array = final_array[final_array[:,1].argsort()]

    # print(sorted_final_array)

    if (subset_array[0][1])<=(sorted_final_array[0][1]):
        print("best KNN regression subset, error, k: " + str(subset_array[0]))
        subset_element = subset_array[0][0]
        subset_min_error = subset_array[0][1]
        subset_min_error_k = subset_array[0][2]
        return [[subset_element],subset_min_error, subset_min_error_k]
    else:
        print("best KNN regression subset, error, k: " + str(sorted_final_array[0]))
        return sorted_final_array[0]
