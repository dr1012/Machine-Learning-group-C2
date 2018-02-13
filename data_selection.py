import numpy as np
import csv
import random

#number of rows in data: 1599

with open('winequality-red.csv', 'r') as csvfile:
        datareader = csv.reader(csvfile, delimiter=';')
        header = next(datareader)
        data = []
       

        for row in datareader:
            row_of_floats = list(map(float, row))
            data.append(row_of_floats)

# data is  of type list
data_as_array = np.array(data)


training_data_list = []
validation_data_list = []
test_data_list = []
count = 0


while(count<(0.6*1599)):
    x = random.choice(data)
    training_data_list.append(x)
    data.remove(x)
    count =  count + 1
while(count>=0.6*1599 and count<0.8*1599):
    y = random.choice(data)
    validation_data_list.append(y)
    data.remove(y)
    count =  count + 1   
while(count>=0.8*1599 and count<1599):
    z = random.choice(data)
    test_data_list.append(z)
    data.remove(z)
    count =  count + 1


training_data = np.array(training_data_list)
validation_data = np.array(validation_data_list)
test_data = np.array(test_data_list) 

#writing the training data to CSV
with open ('training_data.csv', 'w', newline = '') as csvfile2:
    writer = csv.writer(csvfile2, delimiter =',')
    writer.writerow(header)
    for num in range (0,len(training_data)):
        writer.writerow(training_data[num])

#writing the validation data to CSV
with open ('validation_data.csv', 'w', newline = '') as csvfile3:
    writer = csv.writer(csvfile3, delimiter =',')
    writer.writerow(header)
    for num in range (0,len(validation_data)):
        writer.writerow(validation_data[num])

#writing the validation data to CSV
with open ('test_data.csv', 'w', newline = '') as csvfile4:
    writer = csv.writer(csvfile4, delimiter =',')
    writer.writerow(header)
    for num in range (0,len(test_data)):
        writer.writerow(test_data[num])


                

        
       



