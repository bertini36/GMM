# -*- coding: UTF-8 -*-

"""
Dirichlet distribution distribution
"""

import numpy as np
from scipy.special import gammaln

from inference.python.utils import dirichlet_expectation


class Dir:

    def __init__(self, alpha):
        self.alpha = alpha

    def sufficient_statistics(self, k):
        """
        Expectations Dirichlet sufficient statistics computation
            E[log(\pi_{k})] = \Psi(\alpha_{k})
                               - \Psi(\sum_{i=1}^{K}(\alpha_{i}))
        """
        return dirichlet_expectation(self.alpha, k)

    def natural_params(self, k):
        """
        Dirichlet natural parameters
        \alpha_{k} - 1
        """
        return self.alpha[k] - 1

    def log_partition(self):
        """
        Dirichlet log partition
        log(gamma(\Sum_{i=1}^{K}alpha_{i}))
            - \Sum_{i=1}^{K}log(gamma(alpha_{i}))
        """
        return gammaln(np.sum(self.alpha)) - np.sum(gammaln(self.alpha))
