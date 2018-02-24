from polynomial_regression_individual import inidividual_poly_main_function


with open('training-data.csv', 'r') as csvfile:
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