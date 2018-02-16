import csv
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

#used only for visual representation of distribution 
from scipy.stats import norm

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 14:29:04 2018

@author: kai
"""

#load training data into matrix to be used
with open('winequality-red-commas.csv', 'r') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        header = next(datareader)
        data = []
       
        for row in datareader:
            row_of_floats = list(map(float, row))
            data.append(row_of_floats)

        # data is  of type list
        data_array = np.array(data)
        
        
        
def main():
    """
    This file contains code that seeks to explore descriptive statistics about the data set at hand. 
    It investigates measures for central tendency, dispersion and association
    """


#______________________determining the variables' mean_________________________

    
    fig, ax = plt.subplots() 
    
    #create a  list with the variables' mean values
    mean_array = []
    for i in range(12):
        mean = data_array[:,i].mean()
        mean_array.append(mean)

    
    objects = (header[0],header[1],header[2],header[3],header[4],header[5],header[6],header[7],header[8],header[9],header[10],header[11])
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

    plt.show()
        

#__________________determining the variables' dispersion_______________________
    
    
    
   

## read data from a text file. One number per line
#arch = "test/Log(2)_ACRatio.txt"
#datos = []
#for item in open(arch,'r'):
#    item = item.strip()
#    if item != '':
#        try:
#            datos.append(float(item))
#        except ValueError:
#            pass

    fig2, ax = plt.subplots() 

    for i in range(11):
        # best fit of data
        (mu, sigma) = norm.fit(data_array[:,i])
        
        # the histogram of the data
        n, bins, patches = plt.hist(data_array[:,i], 60, normed=1, facecolor='darkblue', alpha=0.75)
        
        # add a 'best fit' line
        y = mlab.normpdf( bins, mu, sigma)
        l = plt.plot(bins, y, 'r--', linewidth=2)
        
        varname = header[i]
        
        #plot
        plt.xlabel(varname)
        plt.ylabel('values of observations')
        plt.title(r'$\mathrm{Histogram\ of\ %s - }\ \mu=%.2f,\ \sigma=%.2f$' %(varname,mu, sigma))
        fig2.savefig('Measure of Dispersion - %s .pdf' %(varname), bbox_inches = 'tight')
        plt.grid(True)
        
        plt.show()
        
    
    
    

    
    


    
    





if __name__ == '__main__':
    # this bit only runs when this script is called from the command line
    main()
 