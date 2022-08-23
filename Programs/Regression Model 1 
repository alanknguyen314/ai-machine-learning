# #######################################################################
#                                                                       # 
# NAME: Alan K. Nguyen                                                  #
# DATE: Aug 2022                                                        #
# PROJECT: Regression Training for House Prices and Area.               #
# FOLDER: Regression                                                    #
#                                                                       # 
#########################################################################

# The training data 

import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

plt.scatter(x_train, y_train, marker='x', c='r')
plt.title("Housing Prices")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')
plt.show()

w = 250
b = 40

def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      y (ndarray (m,)): target values
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb

tmp_f_wb = compute_model_output(x_train, w, b,)


plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

plt.title("Housing Prices")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')

plt.legend()
plt.show()