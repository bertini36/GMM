# -*- coding: UTF-8 -*-

"""
Linear regression using Tensorflow
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

rng = np.random

# Parameters
learning_rate = 0.01
training_epochs = 100

# Training data
train_X = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59,
                      2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
train_Y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53,
                      1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
n_samples = train_X.shape[0]

# Graph input data
X = tf.placeholder('float')
Y = tf.placeholder('float')

# Optimizable parameters with random initialization
weight = tf.Variable(rng.randn(), name='weight')
bias = tf.Variable(rng.randn(), name='bias')

# Linear model
predictions = (X * weight) + bias

# Loss function: Mean Squared Error
loss = tf.reduce_sum(tf.pow(predictions-Y, 2))/(2*n_samples)

# Gradient descent optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
    train_error = sess.run(loss, feed_dict={X: train_X, Y: train_Y})
    print('Train error={}'.format(train_error))

    # Test error
    test_X = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])
    test_error = sess.run(
        tf.reduce_sum(tf.pow(predictions - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})
    print('Test error={}'.format(test_error))

    print('Weight={} Bias={}'.format(sess.run(weight), sess.run(bias)))

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(weight) * train_X
             + sess.run(bias), label='Fitted line')
    plt.legend()
    plt.show()