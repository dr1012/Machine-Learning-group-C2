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


with open('training_data.csv', 'r') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        header = next(datareader)
        training_data = []
     

        for row in datareader:
            row_of_floats = list(map(float, row))
            training_data.append(row_of_floats)

# data is  of type list
trianing_data_as_array = np.array(training_data)



with open('validation_data.csv', 'r') as csvfile1:
        datareader = csv.reader(csvfile1, delimiter=',')
        header = next(datareader)
        validation_data = []
     

        for row in datareader:
            row_of_floats = list(map(float, row))
            validation_data.append(row_of_floats)

# data is  of type list
validation_data_as_array = np.array(validation_data)

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

def getResponse(neighbors):
    #a dictionary containing the different claasses and the number of neighbors that belong to that class
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]
# In case of a  draw this selects one response, ideally we would select an ubiased random response.


#  Simple function that evaluates accuracy of predictions
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0



max_neighbours = 200
accuracies = []
for y in range(1,max_neighbours):

    predictions = []
    for x in range(len(validation_data)):
        neighbors = getNeighbors(training_data, validation_data[x], y)
        result = getResponse(neighbors)
        predictions.append(result)
        #print('> predicted=' + repr(result) + ', actual=' + repr(validation_data[x][-1]))
        
    accuracy = getAccuracy(validation_data, predictions)
    #print('Accuracy: ' + repr(accuracy) + '%')

    accuracies.append([y,accuracy])

npAcurracies = np.array(accuracies)
plt.plot(npAcurracies[:,0:1],npAcurracies[:,1:2])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(npAcurracies[:,0:1],npAcurracies[:,1:2])
ax.set_xlabel("Number of nearest neighbors chosen")
ax.set_ylabel(" % Accuracy of predictions")
ax.set_title("Accuracy of KNN CLASSIFICATION method versus number of nearest neighbors chosen",fontsize = 10)
fig.savefig("KNN classification accuracy.pdf", fmt="pdf")
plt.show()	
