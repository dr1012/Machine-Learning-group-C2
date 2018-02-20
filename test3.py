import numpy as np

total_total_array = []
total_array = []
a = (np.array([[1],[2],[3],[4]]))
b = (np.array([[10],[20],[30],[40]]))
degree = 2
expanded_inputs = []
for k in range(degree+1):
    for j in range (k+1):
        for i in range (k-j+1):
            if i+j == k:
                expanded_inputs.append((a**i)*(b**j))

print(a)
print()
print(b)
print()
expanded_inputs = np.array(expanded_inputs)
expanded_inputs = expanded_inputs.transpose()
expanded_inputs = np.matrix(expanded_inputs)
print(expanded_inputs)

