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
training_epochs = 100

# Training data
train_X = np.array([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59,
                    2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
train_Y = np.array([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53,
                    1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
n_samples = train_X.shape[0]


def loss((weight, bias)):
    """ Loss function: Mean Squared Error """
    predictions = (train_X * weight) + bias
    return np.sum(np.power(predictions - train_Y, 2) / (2 * n_samples))

# Function that returns gradients of loss function
gradient_fun = elementwise_grad(loss)

# Optimizable parameters with random initialization
weight = rng.randn()
bias = rng.randn()

for epoch in range(training_epochs):
    gradients = gradient_fun((weight, bias))
    weight -= gradients[0] * learning_rate
    bias -= gradients[1] * learning_rate
print('Train error={}'.format(loss((weight, bias))))

# Test error
test_X = np.array([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
test_Y = np.array([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])
predictions = (test_X * weight) + bias
print('Test error={}'.format(
    np.sum(np.power(predictions - test_Y, 2) / (2 * n_samples))))

print('Weight={} Bias={}'.format(weight, bias))

# Graphic display
plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.plot(train_X, weight * train_X + bias, label='Fitted line')
plt.legend()
plt.show()
