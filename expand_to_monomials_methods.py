import numpy as np

def expand_to_monomials_2(inputs, degree):


 
    a = (inputs[:][:,0:1])
    b = (inputs[:][:,1:2])
    expanded_inputs = []
    for k in range(degree+1):
        for v in range (k+1):
            for u in range (k-v+1):
                if v+u == k:
                    expanded_inputs.append((a**u)*(b**v))
                     
         
    return np.array(expanded_inputs).transpose()

def expand_to_monomials_3(inputs, degree):


 
    a = (inputs[:][:,0:1])
    b = (inputs[:][:,1:2])
    c = (inputs[:][:,2:3])
    expanded_inputs = []
    for k in range(degree+1):
        for v in range (k+1):
            for u in range (k-v+1):
                for t in range (k-v-u+1):
                    if v+u+t == k:
                        expanded_inputs.append((a**t)*(b**u)*(c**v))
                     
         
    return np.array(expanded_inputs).transpose()


def expand_to_monomials_4(inputs, degree):
  


 
    a = (inputs[:][:,0:1])
    b = (inputs[:][:,1:2])
    c = (inputs[:][:,2:3])
    d = (inputs[:][:,3:4])

    expanded_inputs = []
    for k in range(degree+1):
        for v in range (k+1):
            for u in range (k-v+1):
                for t in range (k-v-u+1):
                    for s in range (k-v-u-t+1):
                        if v+u+t+s == k:
                            expanded_inputs.append((a**s)*(b**t)*(c**u)*(d**v))
                     
         
    return np.array(expanded_inputs).transpose()



def expand_to_monomials_5(inputs, degree):
  
    

 
    a = (inputs[:][:,0:1])
    b = (inputs[:][:,1:2])
    c = (inputs[:][:,2:3])
    d = (inputs[:][:,3:4])
    e = (inputs[:][:,4:5])
    expanded_inputs = []
    for k in range(degree+1):
        for v in range (k+1):
            for u in range (k-v+1):
                for t in range (k-v-u+1):
                    for s in range (k-v-u-t+1):
                        for r in range (k-v-u-t-s+1):
                            if v+u+t+s+r == k:
                                expanded_inputs.append((a**r)*(b**s)*(c**t)*(d**u)*(e**v))
                     
         
    return np.array(expanded_inputs).transpose()

def expand_to_monomials_6(inputs, degree):
  
    

 
    a = (inputs[:][:,0:1])
    b = (inputs[:][:,1:2])
    c = (inputs[:][:,2:3])
    d = (inputs[:][:,3:4])
    e = (inputs[:][:,4:5])
    f = (inputs[:][:,5:6])
    expanded_inputs = []
    for k in range(degree+1):
        for v in range (k+1):
            for u in range (k-v+1):
                for t in range (k-v-u+1):
                    for s in range (k-v-u-t+1):
                        for r in range (k-v-u-t-s+1):
                            for q in range (k-v-u-t-s-r+1):
                                if v+u+t+s+r+q == k:
                                    expanded_inputs.append((a**q)*(b**r)*(c**s)*(d**t)*(e**u)*(f**v))
                     
         
    return np.array(expanded_inputs).transpose()    


def expand_to_monomials_7(inputs, degree):
  


 
    a = (inputs[:][:,0:1])
    b = (inputs[:][:,1:2])
    c = (inputs[:][:,2:3])
    d = (inputs[:][:,3:4])
    e = (inputs[:][:,4:5])
    f = (inputs[:][:,5:6])
    g = (inputs[:][:,6:7])
    expanded_inputs = []
    for k in range(degree+1):
        for v in range (k+1):
            for u in range (k-v+1):
                for t in range (k-v-u+1):
                    for s in range (k-v-u-t+1):
                        for r in range (k-v-u-t-s+1):
                            for q in range (k-v-u-t-s-r+1):
                                for p in range (k-v-u-t-s-r-q+1):
                                    if v+u+t+s+r+q+p == k:
                                        expanded_inputs.append((a**p)*(b**q)*(c**r)*(d**s)*(e**t)*(f**u)*(g**v))
                     
         
    return np.array(expanded_inputs).transpose()



def expand_to_monomials_8(inputs, degree):
  


 
    a = (inputs[:][:,0:1])
    b = (inputs[:][:,1:2])
    c = (inputs[:][:,2:3])
    d = (inputs[:][:,3:4])
    e = (inputs[:][:,4:5])
    f = (inputs[:][:,5:6])
    g = (inputs[:][:,6:7])
    h = (inputs[:][:,7:8])
    expanded_inputs = []
    for k in range(degree+1):
        for v in range (k+1):
            for u in range (k-v+1):
                for t in range (k-v-u+1):
                    for s in range (k-v-u-t+1):
                        for r in range (k-v-u-t-s+1):
                            for q in range (k-v-u-t-s-r+1):
                                for p in range (k-v-u-t-s-r-q+1):
                                    for o in range (k-v-u-t-s-r-q-p+1):
                                        if v+u+t+s+r+q+p+o == k:
                                            expanded_inputs.append((a**o)*(b**p)*(c**q)*(d**r)*(e**s)*(f**t)*(g**u)*(h**v))
                     
         
    return np.array(expanded_inputs).transpose()


def expand_to_monomials_9(inputs, degree):
  


 
    a = (inputs[:][:,0:1])
    b = (inputs[:][:,1:2])
    c = (inputs[:][:,2:3])
    d = (inputs[:][:,3:4])
    e = (inputs[:][:,4:5])
    f = (inputs[:][:,5:6])
    g = (inputs[:][:,6:7])
    h = (inputs[:][:,7:8])
    x = (inputs[:][:,8:9])
    expanded_inputs = []
    for k in range(degree+1):
        for v in range (k+1):
            for u in range (k-v+1):
                for t in range (k-v-u+1):
                    for s in range (k-v-u-t+1):
                        for r in range (k-v-u-t-s+1):
                            for q in range (k-v-u-t-s-r+1):
                                for p in range (k-v-u-t-s-r-q+1):
                                    for o in range (k-v-u-t-s-r-q-p+1):
                                        for n in range (k-v-u-t-s-r-q-p-o+1):
                                                    if v+u+t+s+r+q+p+o+n == k:
                                                        expanded_inputs.append((a**n)*(b**o)*(c**p)*(d**q)*(e**r)*(f**s)*(g**t)*(h**u)*(x**v))
                     
         
    return np.array(expanded_inputs).transpose()
    

def expand_to_monomials_10(inputs, degree):
  


 
    a = (inputs[:][:,0:1])
    b = (inputs[:][:,1:2])
    c = (inputs[:][:,2:3])
    d = (inputs[:][:,3:4])
    e = (inputs[:][:,4:5])
    f = (inputs[:][:,5:6])
    g = (inputs[:][:,6:7])
    h = (inputs[:][:,7:8])
    x = (inputs[:][:,8:9])
    y = (inputs[:][:,9:10])
    expanded_inputs = []
    for k in range(degree+1):
        for v in range (k+1):
            for u in range (k-v+1):
                for t in range (k-v-u+1):
                    for s in range (k-v-u-t+1):
                        for r in range (k-v-u-t-s+1):
                            for q in range (k-v-u-t-s-r+1):
                                for p in range (k-v-u-t-s-r-q+1):
                                    for o in range (k-v-u-t-s-r-q-p+1):
                                        for n in range (k-v-u-t-s-r-q-p-o+1):
                                            for m in range (k-v-u-t-s-r-q-p-o-n+1):
                                                    if v+u+t+s+r+q+p+o+n+m == k:
                                                        expanded_inputs.append((a**m)*(b**n)*(c**o)*(d**p)*(e**q)*(f**r)*(g**s)*(h**t)*(x**u)*(y**v))
                     
         
    return np.array(expanded_inputs).transpose()