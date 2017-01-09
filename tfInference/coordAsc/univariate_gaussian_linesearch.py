# -*- coding: UTF-8 -*-

import math
import numpy as np
import tensorflow as tf

DEBUG = True
SUMMARIES = True
PRECISON = 0.0000001

# Learning rates
#        a     b    mu   beta
lrs = [1e-6, 1e-6, 1e-3, 1e-3]

def printf(s):
    if DEBUG:
        print(s) 

# Data
N = 100
np.random.seed(7)
xn = tf.convert_to_tensor(np.random.normal(5, 1, N), dtype=tf.float64)

m = tf.Variable(0., dtype=tf.float64)
beta = tf.Variable(0.0001, dtype=tf.float64)
a = tf.Variable(0.001, dtype=tf.float64)
b = tf.Variable(0.001, dtype=tf.float64)

# Needed for variational initilizations
a_gamma_ini = np.random.gamma(1, 1, 1)[0]
b_gamma_ini = np.random.gamma(1, 1, 1)[0]

# Variational parameters
a_gamma_var = tf.Variable(a_gamma_ini, dtype=tf.float64)
b_gamma_var = tf.Variable(b_gamma_ini, dtype=tf.float64)
m_mu = tf.Variable(np.random.normal(0., (0.0001)**(-1.), 1)[0], dtype=tf.float64)
beta_mu_var = tf.Variable(np.random.gamma(a_gamma_ini, b_gamma_ini, 1)[0], dtype=tf.float64)

# Maintain numerical stability
a_gamma = tf.nn.softplus(a_gamma_var)
b_gamma = tf.nn.softplus(b_gamma_var)
beta_mu = tf.nn.softplus(beta_mu_var)

LB = tf.mul(tf.cast(1./2, tf.float64), tf.log(tf.div(beta, beta_mu)))
LB = tf.add(LB, tf.mul(tf.mul(tf.cast(1./2, tf.float64), tf.add(tf.pow(m_mu, 2), tf.div(tf.cast(1., tf.float64), beta_mu))), tf.sub(beta_mu, beta)))
LB = tf.sub(LB, tf.mul(m_mu, tf.sub(tf.mul(beta_mu, m_mu), tf.mul(beta, m))))
LB = tf.add(LB, tf.mul(tf.cast(1./2, tf.float64), tf.sub(tf.mul(beta_mu, tf.pow(m_mu, 2)), tf.mul(beta, tf.pow(m, 2)))))

LB = tf.add(LB, tf.mul(a, tf.log(b)))
LB = tf.sub(LB, tf.mul(a_gamma, tf.log(b_gamma)))
LB = tf.add(LB, tf.lgamma(a_gamma))
LB = tf.sub(LB, tf.lgamma(a))
LB = tf.add(LB, tf.mul(tf.sub(tf.digamma(a_gamma), tf.log(b_gamma)), tf.sub(a, a_gamma)))
LB = tf.add(LB, tf.mul(tf.div(a_gamma, b_gamma), tf.sub(b_gamma, b)))

LB = tf.add(LB, tf.mul(tf.div(tf.cast(N, tf.float64), tf.cast(2., tf.float64)), tf.sub(tf.digamma(a_gamma), tf.log(b_gamma))))
LB = tf.sub(LB, tf.mul(tf.div(tf.cast(N, tf.float64), tf.cast(2., tf.float64)), tf.log(tf.mul(tf.cast(2., tf.float64), math.pi))))
LB = tf.sub(LB, tf.mul(tf.cast(1./2, tf.float64), tf.mul(tf.div(a_gamma, b_gamma), tf.reduce_sum(tf.pow(xn, 2)))))
LB = tf.add(LB, tf.mul(tf.div(a_gamma, b_gamma), tf.mul(tf.reduce_sum(xn), m_mu)))
LB = tf.sub(LB, tf.mul(tf.div(tf.cast(N, tf.float64), tf.cast(2., tf.float64)), tf.mul(tf.div(a_gamma, b_gamma), tf.add(tf.pow(m_mu, 2), tf.div(tf.cast(1., tf.float64), beta_mu)))))

# Optimizer definition (Coordinate descent simulation)
mode = tf.placeholder(tf.int32, shape=[], name='mode')
learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
grads_and_vars = None

def f0(): 
    grads_and_vars = optimizer.compute_gradients(-LB, var_list=[a_gamma_var])
    return optimizer.apply_gradients(grads_and_vars)
def f1(): 
    grads_and_vars = optimizer.compute_gradients(-LB, var_list=[b_gamma_var])
    return optimizer.apply_gradients(grads_and_vars)
def f2(): 
    grads_and_vars = optimizer.compute_gradients(-LB, var_list=[m_mu])
    return optimizer.apply_gradients(grads_and_vars)
def f3(): 
    grads_and_vars = optimizer.compute_gradients(-LB, var_list=[beta_mu_var])
    return optimizer.apply_gradients(grads_and_vars)

train = tf.case({tf.cast(mode==0, dtype=tf.bool): f0, 
                 tf.cast(mode==1, dtype=tf.bool): f1, 
                 tf.cast(mode==2, dtype=tf.bool): f2, 
                 tf.cast(mode==3, dtype=tf.bool): f3}, 
                default=f0, exclusive=True)

# Summaries definition
if SUMMARIES:
    tf.summary.histogram('m_mu', m_mu)
    tf.summary.histogram('beta_mu', beta_mu)
    tf.summary.histogram('a_gamma', a_gamma)
    tf.summary.histogram('b_gamma', b_gamma)
    tf.summary.histogram('mode', mode)
    tf.summary.histogram('learning_rate', learning_rate)
    merged = tf.summary.merge_all()
    file_writer = tf.summary.FileWriter('/tmp/tensorboard/', tf.get_default_graph())
    run_calls = 0

def compute_learning_rate(var, alphao):

    grads_and_vars = optimizer.compute_gradients(-LB, var_list=[var])
    grads = sess.run(grads_and_vars)
    tmp_var = grads[0][1]
    tmp_grad = grads[0][0]
   # print "Gradient ", tmp_grad 

    alpha = alphao
    c = 0.1
    tau = 0.5

    or_value = var.eval()

    fx = sess.run(-LB)
    tmp_mod = tmp_var-alpha*tmp_grad
    assign_op = var.assign(tmp_mod)
    _ = sess.run(assign_op)
    fxgrad = sess.run(-LB)

    while np.isinf(fxgrad) or np.isnan(fxgrad):
        alpha /= 10.
        tmp_mod = tmp_var-alpha*tmp_grad
        assign_op = var.assign(tmp_mod)
        _ = sess.run(assign_op)
        fxgrad = sess.run(-LB)


    m = tmp_grad

    #print "alpha:", alpha, "fx", fx,"fxgrad:", fxgrad,  "alpha*c*m", alpha*c*m
    while (fxgrad >= fx + alpha*c*m):
        alpha *= tau
        tmp_mod = tmp_var-alpha*tmp_grad
        assign_op = var.assign(tmp_mod)
        _ = sess.run(assign_op)
        fxgrad = sess.run(-LB)

        #grads_and_vars = optimizer.compute_gradients(-LB, var_list=[var])
        #aux_grads = sess.run(grads_and_vars)
        #m = tmp_grad*aux_grads[0][0]

        #print "alpha:", alpha, "fx", fx,"fxgrad:", fxgrad,  "alpha*c*m", alpha*c*m
        if alpha < 1e-10:
            alpha = 0
            break

    return alpha, tmp_var



# Main program

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(100):
        # printf('***** Epoch {} *****'.format(epoch))

        alphao = 1e10

        alpha, tmp_var = compute_learning_rate(a_gamma_var, alphao)
        #print tmp_var
        assign_op = a_gamma_var.assign(tmp_var)
        _ = sess.run(assign_op)
        #print "a_gamma_var", a_gamma_var.eval(sess)
        #print "b_gamma_var", b_gamma_var.eval(sess)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
        grads_and_vars = optimizer.compute_gradients(-LB, var_list=[a_gamma_var])
        train = optimizer.apply_gradients(grads_and_vars)
        _, lb, grads, a = sess.run([train, LB, grads_and_vars, a_gamma])
        alpha1 = alpha
        #print alpha*grads[0][0]
        #print grads
        #printf('a_gamma_var: value={}->{}  alpha= {} gradient={}'.format(tmp_var, grads[0][1], alpha, grads[0][0]))
        # print "a_gamma_var", a_gamma_var.eval(sess)
        # print "b_gamma_var", b_gamma_var.eval(sess)
        # print "a_gamma b_gamma", sess.run([a_gamma, b_gamma])

        alpha, tmp_var = compute_learning_rate(b_gamma_var, alphao)
        assign_op = b_gamma_var.assign(tmp_var)
        _ = sess.run(assign_op)   
        # print "a_gamma_var", a_gamma_var.eval(sess)
        # print "b_gamma_var", b_gamma_var.eval(sess)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
        grads_and_vars = optimizer.compute_gradients(-LB, var_list=[b_gamma_var])
        train = optimizer.apply_gradients(grads_and_vars)
        _, lb, grads, b = sess.run([train, LB, grads_and_vars, b_gamma])
        alpha2 = alpha
        #print alpha*grads[0][0]
        #printf('b_gamma_var: value={}->{} alpha= {}  gradient={}'.format(tmp_var, grads[0][1],  alpha, grads[0][0]))
        # printf('b_gamma:{}'.format(b))
        # print "a_gamma_var", a_gamma_var.eval(sess)
        # print "b_gamma_var", b_gamma_var.eval(sess)
        # print "a_gamma b_gamma", sess.run([a_gamma, b_gamma])

        alpha, tmp_var = compute_learning_rate(m_mu, alphao)
        #print tmp_var
        assign_op = m_mu.assign(tmp_var)
        _ = sess.run(assign_op)      
        # print "a_gamma_var", a_gamma_var.eval(sess)
        # print "b_gamma_var", b_gamma_var.eval(sess)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
        grads_and_vars = optimizer.compute_gradients(-LB, var_list=[m_mu])
        train = optimizer.apply_gradients(grads_and_vars)
        _, lb, grads = sess.run([train, LB, grads_and_vars])
        alpha3 = alpha
        #print alpha*grads[0][0]
        #printf('m_mu: value={}->{} alpha= {} gradient={}'.format( tmp_var, grads[0][1], alpha, grads[0][0]))
        # print "a_gamma_var", a_gamma_var.eval(sess)
        # print "b_gamma_var", b_gamma_var.eval(sess)
        # print "a_gamma b_gamma", sess.run([a_gamma, b_gamma])

        alpha, tmp_var = compute_learning_rate(beta_mu_var, alphao)
        #print tmp_var
        assign_op = beta_mu_var.assign(tmp_var)
        _ = sess.run(assign_op)      
        # print "a_gamma_var", a_gamma_var.eval(sess)
        # print "b_gamma_var", b_gamma_var.eval(sess)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
        grads_and_vars = optimizer.compute_gradients(-LB, var_list=[beta_mu_var])
        train = optimizer.apply_gradients(grads_and_vars)
        _, lb, grads, mu, beta, a, b = sess.run([train, LB, grads_and_vars,  m_mu, beta_mu, a_gamma, b_gamma])
        alpha4 = alpha
        # print "a_gamma_var", a_gamma_var.eval(sess)
        # print "b_gamma_var", b_gamma_var.eval(sess)
        # print "a_gamma b_gamma", sess.run([a_gamma, b_gamma])
        # print alpha*grads[0][0]
        #printf('beta_mu: value={}->{} alpha= {} gradient={}'.format( tmp_var, grads[0][1], alpha, grads[0][0]))

        # print "a_gamma_var", a_gamma_var.eval(sess)
        # print "b_gamma_var", b_gamma_var.eval(sess)

        #print alpha1, alpha2, alpha3, alpha4
        if epoch > 0:
            inc = (old_ELBO-lb)/old_ELBO*100
            printf('It={}, ELBO={}, Inc={}, Mean={}, beta={}, Precision={}, a={}, b={}'.format(epoch, lb, inc, mu, beta, a/b, a, b))
            if inc < 1e-8:
                break;
        else:
            printf('It={}, ELBO={}, Inc={}, Mean={}, beta={}, Precision={}, a={}, b={}'.format(epoch, lb, None, mu, beta, a/b, a, b))

        old_ELBO = lb.copy()
