import csv
import math
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib import cm as cm
import pandas as pd
from pandas.plotting import scatter_matrix

#used only for visual representation of distribution
from scipy.stats import norm

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 14:29:04 2018

@author: kai
"""


#load data into matrix to be used
with open('final_training_data.csv', 'r') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        header = next(datareader)
        data = []

        for row in datareader:
            row_of_floats = list(map(float, row))
            data.append(row_of_floats)

        # data is  of type list
        data_array = np.array(data)


def mainStat():
    """
    This file contains code that seeks to explore descriptive statistics about the data set at hand.
    It investigates measures for central tendency, dispersion and association
    """

    # determining the variables' mean
    explore_var_means()

    #determining the variables' dispersion
    explore_var_dispersion()

    #determining the variables' association
    explore_var_association()

    #explore possible outliers
    explore_data_outliers()

    #explore scatter plot
    explore_scatter()


def explore_var_means():
    """
    Explores the mean of the variables
    """

    fig, ax = plt.subplots()

    #create a  list with the variables' mean values
    mean_array = []
    for i in range(12):
        mean = data_array[:,i].mean()
        mean_array.append(mean)

    #store the variable names in the objects array
    objects = []
    for i in range(len(header)):
        objects.append(header[i])

    print(objects)
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, mean_array, align='center', alpha=0.5)
    plt.xticks(y_pos, objects, rotation=90)

    plt.ylabel('Mean value')
    plt.yticks(np.arange(0, max(mean_array)+15, 5))
    # create a list to collect the plt.patches data
    totals = []
    # find the values and append to list
    for i in ax.patches:
        totals.append(i.get_height())
    # set individual bar lables using above list
    total = sum(totals)
    # set individual bar lables using above list
    for i in ax.patches:
        # get_x pulls left or right; get_height pushes up or down
        ax.text(i.get_x()+.17, i.get_height()+10, \
                str(round((i.get_height()/total)*100, 2)), fontsize=12,
                    color='dimgrey', rotation=90)

    plt.title('Measure of Central Tendency: variable means')
    fig.savefig('Measure of Central Tendency - variable means.pdf', bbox_inches = 'tight')
    # plt.show()


def explore_var_dispersion():
    """
    Explores the dispersion of the variables
    """

    for i in range(len(data_array[0])):

        fig2, ax = plt.subplots()
        # best fit of data
        (mu, sigma) = norm.fit(data_array[:,i])

        # the normalized histogram of the data
#        n, bins, patches = plt.hist(data_array[:,i], 60, normed=1,  facecolor='darkblue', alpha=0.75)
#        the non-normalized histogram of the data
        n, bins, patches = plt.hist(data_array[:,i], 60, facecolor='darkblue', alpha=0.75)

        # add a 'best fit' line
        y = mlab.normpdf( bins, mu, sigma)
        plt.plot(bins,y, 'r--', linewidth=1)

        varname = header[i]

        #plotting the histogram
        plt.xlabel(varname)
#        plt.ylabel('Normalized Frequency')
        plt.ylabel('Frequency')
#        plt.title(r'$\mathrm{Normalized Histogram\ of\ %s - }\ \mu=%.2f,\ \sigma=%.2f$' %(varname,mu, sigma))
        plt.title(r'$\mathrm{Histogram\ of\ %s - }\ \mu=%.2f,\ \sigma=%.2f$' %(varname,mu, sigma))
#        fig2.savefig('Measure of Dispersion Normalized - %s .pdf' %(varname), bbox_inches = 'tight')
        fig2.savefig('Measure of Dispersion - %s .pdf' %(varname), bbox_inches = 'tight')
        # plt.show()

def explore_var_association():
    """
    Explores the association of the variables
    """

    df = pd.read_csv('winequality-red.csv',delimiter=';')
#    df = pd.DataFrame.as_matrix(data_array)
    fig3, ax = plt.subplots()
    plt.matshow(df.corr(method='pearson'), interpolation="nearest")
    plt.xticks(range(len(df.columns)), df.columns,fontsize=10, rotation=90)
    plt.yticks(range(len(df.columns)), df.columns,fontsize=10)
    plt.colorbar()
    plt.suptitle('Variable Correlation Matrix', y= 0.1, x = .375)
    plt.savefig("Variable Correlation Matrix.pdf", bbox_inches='tight')


def explore_data_outliers():
    """
    Explores if there are any outliers in the data
    """

    fig, ax = plt.subplots()

    #log rescale data to fit into one boxplot figure
    data = []
    for i in range(12):
        data.append(data_array[:,i])

    #plot the boxplots
    plt.ylim((-5,6.5))
    ax.boxplot(data, sym='.')
    plt.xlabel('dataset variables')
    plt.ylabel('log(values)')
    plt.title('outlier analysis - boxplots of all variables')
    plt.xticks(range(len(header)+1), (' ','fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality'), rotation=90)
    fig.savefig('Boxplots.pdf', bbox_inches = 'tight')
    # plt.show()


def explore_scatter():
    """
    Creates a scotter plot of all variables in the data set to explore potential
    relationships between the variablesgit
    """


    df = pd.DataFrame(data, columns=header)
    sm = scatter_matrix(df, alpha=0.2, figsize=(10, 10), diagonal='kde')

    [s.xaxis.label.set_rotation(45) for s in sm.reshape(-1)]
    [s.yaxis.label.set_rotation(45) for s in sm.reshape(-1)]

    #May need to offset label when rotating to prevent overlap of figure
    [s.get_yaxis().set_label_coords(-1,0.5) for s in sm.reshape(-1)]

    #Hide all ticks
    [s.set_xticks(()) for s in sm.reshape(-1)]
    [s.set_yticks(()) for s in sm.reshape(-1)]
    #    plt.tight_layout(w_pad=.01)

    plt.title("Scatter plots - All variables",  y=12.5, x = -5.5)
    plt.savefig("scatterplot.pdf")
    # plt.show()


if __name__ == '__main__':
    # this bit only runs when this script is called from the command line
    mainStat()
