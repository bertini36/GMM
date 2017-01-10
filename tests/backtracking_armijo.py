# -*- coding: UTF-8 -*-

import math
import numpy as np
import tensorflow as tf

x = tf.Variable(-10,dtype=tf.float64)
f = tf.mul(tf.add(x,tf.constant(5,dtype=tf.float64)),tf.add(x,tf.constant(5,dtype=tf.float64)))


def compute_learning_rate(var, alphao):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alphao)
    grads_and_vars = optimizer.compute_gradients(f, var_list=[var])
    grads = sess.run(grads_and_vars)
    tmp_var = grads[0][1]
    tmp_grad = grads[0][0]
   # print "Gradient ", tmp_grad 

    alpha = alphao
    c = 0.5
    tau = 0.2

    or_value = var.eval()

    fx = sess.run(f)
    tmp_mod = tmp_var-alpha*tmp_grad
    assign_op = var.assign(tmp_mod)
    _ = sess.run(assign_op)
    fxgrad = sess.run(f)

    while np.isinf(fxgrad) or np.isnan(fxgrad):
        alpha /= 10.
        tmp_mod = tmp_var-alpha*tmp_grad
        assign_op = var.assign(tmp_mod)
        _ = sess.run(assign_op)
        fxgrad = sess.run(f)


    m = tmp_grad**2

    #print "alpha:", alpha, "fx", fx,"fxgrad:", fxgrad,  "alpha*c*m", alpha*c*m
    while (fxgrad >= fx -alpha*c*m):
        alpha *= tau
        tmp_mod = tmp_var-alpha*tmp_grad
        assign_op = var.assign(tmp_mod)
        _ = sess.run(assign_op)
        fxgrad = sess.run(f)

        #grads_and_vars = optimizer.compute_gradients(-LB, var_list=[var])
        #aux_grads = sess.run(grads_and_vars)
        #m = tmp_grad*aux_grads[0][0]

        #print "alpha:", alpha, "fx", fx,"fxgrad:", fxgrad,  "alpha*c*m", -alpha*c*m
        if alpha < 1e-10:
            alpha = 0
            break

    return alpha, tmp_var, tmp_grad


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for it in xrange(100):
        alpha, tmp_var, tmp_grad =  compute_learning_rate(x, 100)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
        grads_and_vars = optimizer.compute_gradients(f, var_list=[x])
        F, grads = sess.run([f, grads_and_vars])


        if it > 0:
            diff = (F - old_F)
            print('It={}, F={}, Diff={}, x={}'.format(it, F, diff, grads[0][1]))
            if diff > - 1e-4:
                break;
        else:
            print('It={}, F={}, Inc={}, x={}'.format(it, F, None, grads[0][1]))

        old_F = F
