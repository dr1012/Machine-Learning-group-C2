import csv
import numpy as np
import random


def main(ifname, delimiter=None, columns=None, has_header=True):

    # if no file name is provided then use synthetic data
    import_data(
            ifname, delimiter=delimiter, has_header=has_header, columns=columns)


def import_data(ifname,delimiter=None, has_header=False, columns=None):
    """
    Imports a tab/comma/semi-colon/... separated data file as an array of
    floating point numbers. If the import file has a header then this should
    be specified, and the field names will be returned as the second argument.

    parameters
    ----------
    ifname -- filename/path of data file.
    delimiter -- delimiter of data values
    has_header -- does the data-file have a header line
    columns -- a list of integers specifying which columns of the file to import
        (counting from 0)

    returns
    -------
    data_as_array -- the data as a numpy.array object
    field_names -- if file has header, then this is a list of strings of the
      the field names imported. Otherwise, it is a None object.
    """
    # delimiter = ","
    with open(ifname, 'r') as ifile:
        datareader = csv.reader(ifile, delimiter=delimiter)
        # if the data has a header line we want to avoid trying to import it.
        # instead we'll print it to screen
        if has_header:
            headers = next(datareader)
            print("Importing data with field_names:\n\t" + ",".join(headers))
        else:
            # if there is no header then the field names is a dummy variable
            headers = None
        # create an empty list to store each row of data
        data = []
        for row in datareader:
            # for each row of data only take the columns we are interested in
            if not columns is None:
                row = [row[c] for c in columns]
            # now store in our data list
            data.append(row)
        print("There are %d entries" % len(data))
        print("Each row has %d elements" % len(data[0]))
    # convert the data (list object) into a numpy array.
    data_as_array = np.array(data).astype(float)
    if not columns is None and not headers is None:
        # thin the associated field names if needed
        headers = [headers[c] for c in columns]
    # return this array to caller (and field_names if provided)

    n = len(data)

    data_list = data[:]
    training_data_list = []
    test_data_list = []
    count = 0

    while (count < (0.9 * n)):
        x = random.choice(data)
        training_data_list.append(x)
        data.remove(x)
        count = count + 1
    while (count >= 0.9 * n and count < n):
        z = random.choice(data)
        test_data_list.append(z)
        data.remove(z)
        count = count + 1

    training_data = np.array(training_data_list)
    test_data = np.array(test_data_list)

    # writing the validation data to CSV
    with open('entire_data.csv', 'w') as csvfile1:
        writer = csv.writer(csvfile1, delimiter=',')
        writer.writerow(headers)
        for num in range(0, len(data_list)):
            writer.writerow(data_list[num])

    # writing the training data to CSV
    with open('final_training_data.csv', 'w') as csvfile2:
        writer = csv.writer(csvfile2, delimiter=',')
        writer.writerow(headers)
        for num in range(0, len(training_data)):
            writer.writerow(training_data[num])

    # writing the validation data to CSV
    with open('final_test_data.csv', 'w') as csvfile3:
        writer = csv.writer(csvfile3, delimiter=',')
        writer.writerow(headers)
        for num in range(0, len(test_data)):
            writer.writerow(test_data[num])

if __name__ == '__main__':
    """
    To run this script on just synthetic data use:

        python regression_external_data.py

    You can pass the data-file name as the first argument when you call
    your script from the command line. E.g. use:

        python regression_external_data.py datafile.tsv

    If you pass a second argument it will be taken as the delimiter, e.g.
    for comma separated values:

        python regression_external_data.py comma_separated_data.csv ","

    for semi-colon separated values:

        python regression_external_data.py comma_separated_data.csv ";"

    If your data has more than 2 columns you must specify which columns
    you wish to plot as a comma separated pair of values, e.g.

        python regression_external_data.py comma_separated_data.csv ";" 8,9

    For the wine quality data you will need to specify which columns to pass.
    """
    import sys
    if len(sys.argv) == 1:
        main()  # calls the main function with no arguments
    elif len(sys.argv) == 2:
        # assumes that the first argument is the input filename/path
        main(ifname=sys.argv[1], delimiter=",")
    elif len(sys.argv) == 3:
        # assumes that the second argument is the data delimiter
        # assumes that the third argument is the list of columns to import
        columns = list(map(int, sys.argv[3].split(",")))
        main(ifname=sys.argv[1], delimiter=",", columns=columns)
