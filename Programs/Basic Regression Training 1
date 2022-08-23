#########################################################################
#                                                                       # 
# NAME: Alan K. Nguyen                                                  #
# DATE: Aug 2022                                                        #
# PROJECT: Basic Regression Training Model 1 (Coursera)                 #
# FOLDER: Machine Learning, Regression                                  #
#                                                                       # 
#########################################################################

import numpy as np
import matplotlib.pyplot as plt

# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

# show data

print(f"x_train = {x_train}")
print(f"y_train = {y_train}")



# m is the number of training examples

print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")

"""

# m is the number of training examples (same as above)
m = len(x_train)
print(f"Number of training examples is: {m}")

"""

# TRAINING PROCESS

i = 0 
# Change this to 1 to see (x^1, y^1)
# i here is the index of the data. i = 0 is the first data point.

x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")

# PLOTTING PROCESS

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.show()

# CHANGING VALUES OF W AND B.
# FUNCTION FOR COMPUTING OUTPUT

w = 100
b = 100
print(f"w: {w}")
print(f"b: {b}")

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

# PLOT OUTPUT

tmp_f_wb = compute_model_output(x_train, w, b,)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()

# TEST OUTPUT W AND B TO PREDICT S0METHING

w = 200                         
b = 100    
x_i = 1.2
cost_1200sqft = w * x_i + b    

print(f"${cost_1200sqft:.0f} thousand dollars") # returns $340 Thousands Dollars.








