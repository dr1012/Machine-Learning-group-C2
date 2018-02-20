import numpy as np
import csv
import math
import operator
import matplotlib.pyplot as plt

with open('training_data.csv', 'r') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        header = next(datareader)
        training_data = []
     

        for row in datareader:
            row_of_floats = list(map(float, row))
            training_data.append(row_of_floats)

# data is  of type list
trianing_data_as_array = np.array(training_data)

mysubset = trianing_data_as_array[:][:,[1,2,9,10,11]]
mylist = mysubset.tolist()



N = trianing_data_as_array.shape[0]

centres = trianing_data_as_array[np.random.choice([False,True], size=N, p=[0.9,0.1]),:]

print(centres)


