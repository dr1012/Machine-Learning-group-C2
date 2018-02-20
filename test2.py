import numpy as np
import csv
import math
import operator
import matplotlib.pyplot as plt


import numpy.random as random
import numpy.linalg as linalg


with open('training_data.csv', 'r') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        header = next(datareader)
        training_data = []
     

        for row in datareader:
            row_of_floats = list(map(float, row))
            training_data.append(row_of_floats)

# data is  of type list
trianing_data_as_array = np.array(training_data)


a = (np.array([[1],[2],[3],[4],[5],[6],[7],[8]]))
b = a.transpose()

c = (trianing_data_as_array[:][0:20,0:1]).transpose()

print(a)
print("#############################################")
print(b)
print("#############################################")
print("#############################################")
print(np.matrix(a) * np.matrix(b))
print(np.matrix(b) * np.matrix(a))

print(a**2)




