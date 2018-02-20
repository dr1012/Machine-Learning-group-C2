import numpy as np
import csv
import math
import operator
import matplotlib.pyplot as plt

# loading all the data from the csv file
with open('winequality-red - commas.csv', 'r') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        header = next(datareader)
        all_data = []
     

        for row in datareader:
            row_of_floats = list(map(float, row))
            all_data.append(row_of_floats)

all_data_as_array = np.array(all_data)



# these are lists that will contain the data for each inidividual wine quality (1.0-10.0 hence the names of the lists)
one = []
two = []
three = []
four = []
five =  []
six = []
seven = []
eight = []
nine = []
ten = []

#this for loop goes through all the data from the csv file and adds the data to the relevant lists depending on wine quality value
for x in range(0,len(all_data)):
    if all_data[x][11] == 1.0:
        one.append(all_data[x]) 
    if all_data[x][11] == 2.0:
        two.append(all_data[x]) 
    if all_data[x][11] == 3.0:
        three.append(all_data[x]) 
    if all_data[x][11] == 4.0:
        four.append(all_data[x]) 
    if all_data[x][11] == 5.0:
        five.append(all_data[x]) 
    if all_data[x][11] == 6.0:
        six.append(all_data[x]) 
    if all_data[x][11] == 7.0:
        seven.append(all_data[x]) 
    if all_data[x][11] == 8.0:
        eight.append(all_data[x]) 
    if all_data[x][11] == 9.0:
        nine.append(all_data[x]) 
    if all_data[x][11] == 10.0:
        ten.append(all_data[x]) 



# converting the lists to numpy arrays
one = np.array(one)
two =  np.array(two)
three = np.array(three)
four = np.array(four)
five = np.array(five)
six = np.array(six)
seven = np.array(seven)
eight = np.array(eight)
nine = np.array(nine)
ten = np.array(ten)

# a list of the numpy arrays
categorised_data = [one,two,three,four,five,six,seven,eight,nine,ten]

#for each attribute (PH, Alcohol, residual sugar, etc.) find the mean attribute value for each discrete wine quality and
# plot the quality against the mean attribute
'''
for x in range(0,11):
    means = []
    count = 1.0
    for y in categorised_data:
        if y.size > 0:
            calcmean = np.mean(y[:,x:x+1])
            means.append([calcmean,count])
       
        
        count = count + 1
        
    npMeans = np.array(means)
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(npMeans[:,0:1],npMeans[:,1:2])
    ax.set_xlabel("mean of " + header[x] + " for each quality")
    ax.set_ylabel("Wine Quality")
    fig.savefig("mean of " + header[x]+".pdf", fmt="pdf")
    plt.show()	

  '''





#plot all the raw data for each attribute without going through all of the above
#this doesn't show much 
for x in range(0,11):
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(all_data_as_array[:,x:x+1],all_data_as_array[:,11:12])
    ax.set_xlabel(header[x])
    ax.set_ylabel("Wine Quality")
    ax.set_title("Wine Quality vs " + header[x] ,fontsize = 10)
    fig.savefig(header[x]+".pdf", fmt="pdf")
    plt.show()
    

    



