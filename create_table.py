import sys
import csv
import random
import numpy as np
import matplotlib.pyplot as plt


from tabulate import tabulate

def drawTable(headers, data):

    # headers = ["","LR","Polynomial","RBF","kNN"]
    #
    # data = [["RSE","6.3", "6.1", "6.2","6.5"]]
    print("\n")

    print(tabulate(data,headers=headers,tablefmt="grid"))

# drawTable()