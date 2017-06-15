# -*- coding: UTF-8 -*-

"""
Normal Inverse Wishart distribution
"""

import numpy as np
from numpy.linalg import det, inv
from scipy.special import multigammaln, psi


class NIW:

    def __init__(self, m, beta, nu, w):
        self.m = m                          # (D)
        self.beta = beta                    # ()
        self.nu = nu                        # () > D
        self.w = w                          # (D, D)
        self.D = len(self.m)

    def sufficient_statistics(self):
        """
        Expectations Normal Inverse Wishart sufficient statistics computation
            E[\Sigma^{-1}\mu] = nu * W^{-1} * m
            E[-1/2\Sigma^{-1}] = -1/2 * nu * W^{-1}
            E[-1/2\mu.T\Sigma^{-1}\mu] = -D/2 * beta^{-1} - nu * m.T * W^{-1} *m
            E[-1/2log(|\Sigma|)] = D/2 * log(2) + 1/2 *
                            sum_{i=1}^{D}(Psi(nu/2 + (1-i)/2)) - 1/2 * log(|W|)
        """
        inv_lambda_w = inv(self.w)
        return np.array([
            np.dot(self.nu * inv_lambda_w, self.m),
            (-1 / 2.) * self.nu * inv_lambda_w,
            (-self.D / 2.) * (1 / self.beta) - (1 / 2.) * self.nu *
            np.dot(np.dot(self.m.T, inv_lambda_w), self.m),
            (self.D / 2.) * np.log(2.) + (1 / 2.) *
            np.sum([psi((self.nu / 2.) + ((1 - i) / 2.))
                    for i in range(self.D)])
            - (1 / 2.) * np.log(det(self.w))
        ])

    def natural_params(self):
        """
        Normal Inverse Wishart natural parameters
        m.T * beta
        W + beta * m * m.T
        beta
        nu - D - 2
        """
        return np.array([
            self.beta * self.m,
            self.w + self.beta * np.outer(self.m, self.m),
            self.beta,
            self.nu + self.D + 2
        ])

    def log_partition(self):
        """
        Normal Inverse Wishart log partition
        (nu * D * log(2)) / 2 + log(gamma_{D}(nu/2))
            - D/2 * log(|beta|) - nu/2 * log(|W|)
        """
        return (self.nu * self.D * np.log(2)) / 2. \
               + np.log(multigammaln(self.nu/2., self.D)) \
               - self.D / 2. * np.log(np.absolute(self.beta)) \
               - self.nu / 2. * np.log(det(self.W))
