import numpy as np
import itertools
'''
a = [[1, 1, 12], [2, 3,5], [3, 5, 400], [10,20,10000]]

print(a)

b = np.array(a)

print(b)

sorted_array = b[b[:,2].argsort()]

print(sorted_array)

print(b.shape)

c = [1.0,2.0,3.0,4.0]

d = [int(i) for i in c]

print(d)
'''

def function_1(inputs,degree):
    print(inputs+degree)


a = 1
def function_2(inputs1, degree1):
    eval("function_"+str(a)+"(inputs1,degree1)")    

subset_attributes =  [1,2,3,4,5,6]

count = 0
for L in range(0, len(subset_attributes)+1):
    for subset in itertools.combinations(subset_attributes, L):
#every subset here is a tuple containing the indexes of the attributes in the subset
        if(len(subset)>1):
            print(subset)
            count = count + 1

print(count)

del subset_attributes[-1]

print(subset_attributes)