import matplotlib.pyplot as plt
from scipy import sparse
import numpy as np
import pandas as pd

'''matplotlib
'''
x=np.arange(20)
y=np.sin(x)
print(x)
print(y)
plt.plot(x,y,marker="x")
plt.show()

'''scipy
'''
# create a 2d numpy array with a diagonal of ones, and zeros everywhere else
# only the none-zero entries are stored
eye=np.eye(4)
print('numpy array:\n%s' % eye)
# conver the numpy array to a scipy sparse matrix in CSR format
sparse_matrix=sparse.csr_matrix(eye)
print("\nScipy sparse csr matrix:\n$s" % sparse_matrix)

'''pandas
'''
data={'data':['John','anna','peter','linda'],
'location':['new york','paris','berlin','london'],
'age':[24,23,21,22]}
data_pandas=pd.DataFrame(data)
print(data_pandas)

array=np.arange(20).reshape(4,5)
print(array)
print(array[:,0]) # column 1 in matrix
print(array[:,1]) # column 2
