# -*- coding: UTF-8 -*-

"""
Normal Inverse Wishart distribution
"""

import numpy as np
from numpy.linalg import det, inv
from scipy.special import psi


class NIW:

    def __init__(self, m, beta, nu, w):
        self.m = m
        self.beta = beta
        self.nu = nu
        self.w = w

    def sufficient_statistics(self, k, D):
        """
        Expectations Normal Inverse Wishart sufficient statistics computation
            E[\Sigma^{-1}\mu] = \nu * W^{-1} * m
            E[-1/2\Sigma^{-1}] = -1/2 * \nu * W^{-1}
            E[-1/2\mu.T\Sigma^{-1}\mu] = -D/2 * \beta^{-1} - \nu * m.T * W^{-1} * m
            E[-1/2log|\Sigma|] = D/2 * log(2) + 1/2 *
                sum_{i=1}^{D}(Psi(\nu/2 + (1-i)/2)) - 1/2 * log(|W|)
        """
        inv_lambda_w = inv(self.w[k, :, :])
        return np.array([
            np.dot(self.nu[k] * inv_lambda_w, self.m[k, :]),
            (-1 / 2.) * self.nu[k] * inv_lambda_w,
            (-D / 2.) * (1 / self.beta[k]) - (1 / 2.) * self.nu[k] *
            np.dot(np.dot(self.m[k, :].T, inv_lambda_w), self.m[k, :]),
            (D / 2.) * np.log(2.) + (1 / 2.) *
            np.sum([psi((self.nu[k] / 2.) + ((1 - i) / 2.)) for i in range(D)])
            - (1 / 2.) * np.log(det(self.w[k, :, :]))
        ])

    def natural_params(self, k, D):
        """
        Normal Inverse Wishart natural parameters
        m.T * beta
        W + beta * m * m.T
        beta
        nu - D - 2
        """
        return np.array([
            self.beta[k] * self.m[k, :],
            (self.w[k, :, :] + self.beta[k] *
             np.outer(self.m[k, :], self.m[k, :])),
            self.beta[k],
            self.nu[k] + D + 2
        ])
