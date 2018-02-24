import numpy as np
import numpy.linalg as linalg



from expand_to_monomials_methods import expand_to_monomials_2
from expand_to_monomials_methods import expand_to_monomials_3
from expand_to_monomials_methods import expand_to_monomials_4
from expand_to_monomials_methods import expand_to_monomials_5
from expand_to_monomials_methods import expand_to_monomials_6
from expand_to_monomials_methods import expand_to_monomials_7
from expand_to_monomials_methods import expand_to_monomials_8
from expand_to_monomials_methods import expand_to_monomials_9
from expand_to_monomials_methods import expand_to_monomials_10


def ml_weights(inputmtx, targets):
   
    Phi = np.matrix(inputmtx)
    targets = np.matrix(targets).reshape((len(targets),1))
    weights = linalg.inv(Phi.transpose()*Phi)*Phi.transpose()*targets
    return np.array(weights).flatten()

def construct_polynomial_approx(length,degree,weights):

    
   
    def prediction_function(xs):
        expanded_xs = np.matrix(eval("expand_to_monomials_"+str(length)+"(xs,"+str(degree)+")"))
        ys = expanded_xs*np.matrix(weights).reshape((len(weights),1))
        return np.array(ys).flatten()
   
    return prediction_function 

def function_return(training_data, train_targets, output_array ):
    subset = output_array[0]
    length = len(subset)
    training_subset  = training_data[:][:,subset]
    degree = output_array[0][2]
    processed_inputs = eval("expand_to_monomials_"+str(length)+"(training_subset, degree)")
    weights = ml_weights(processed_inputs, train_targets)
    approx_func = construct_polynomial_approx(length,degree,weights)
    return approx_func