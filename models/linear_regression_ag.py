# -*- coding: UTF-8 -*-

"""
Linear regression using Autograd
"""

import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import elementwise_grad

rng = np.random

# Parameters
learning_rate = 0.01
training_epochs = 1000

# Training data
train_X = np.array([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59,
                    2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
train_Y = np.array([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53,
                    1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])

# Test data
test_X = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1, 3.21,
                     7.01, 2.221, 3.8, 4.43, 2.67, 1.11, 8.88, 5.61])
test_Y = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03, 2.02,
                     5.09, 1.98, 5.48, 2.76, 4.45, 9.89, 3.56, 8.78])

N = train_X.shape[0]


def loss((weight, bias)):
    """ Loss function: Mean Squared Error """
    predictions = (train_X * weight) + bias
    return np.sum(np.power(predictions - train_Y, 2)) / N

# Function that returns gradients of loss function
gradient_fun = elementwise_grad(loss)

# Optimizable parameters with random initialization
weight = rng.randn()
bias = rng.randn()

for epoch in range(training_epochs):
    gradients = gradient_fun((weight, bias))
    weight -= gradients[0] * learning_rate
    bias -= gradients[1] * learning_rate

# Train error
print('Train error={}'.format(loss((weight, bias))))

# Test error
predictions = (test_X * weight) + bias
print('Test error={}'.format(np.sum(np.power(predictions - test_Y, 2)) / N))

# Optimization results
print('Weight={} Bias={}'.format(weight, bias))

# Graphic display
plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.plot(train_X, weight * train_X + bias, label='Fitted line')
plt.legend()
plt.show()
