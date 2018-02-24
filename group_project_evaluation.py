import numpy as np 

from data_import import import_data

#from polynomial_global import main_polynomial_function
from regression_rbf_kai import run_rbf_model



def main(ifname, delimiter=None, columns=None, has_header=True):
    """
    Runs the scripts of the model training and evaluation methods imported
 
    """

    print("___________________I was here :D  ______________________________________")

    #imports the data and produces the training, testing and total csv files
    import_data(ifname, delimiter=delimiter, has_header=has_header, columns=columns)

    #runs the polynomial optimisation model
#    best_polynomial_function = main_polynomial_function()        

#    best_rbf_function = 

    #runs the polynomial optimisation model
#    run_rbf_model()


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