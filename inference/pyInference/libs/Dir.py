# -*- coding: UTF-8 -*-

"""
Dirichlet distribution distribution
"""

from common import dirichlet_expectation


class Dir:

    def __init__(self, pi):
        self.pi = pi

    def sufficient_statistics(self, k):
        """
        Expectations Dirichlet sufficient statistics computation
        E[log(\pi_{k})] = \Psi(\alpha_{k}) - \Psi(\sum_{i=1}^{K}(\alpha_{i}))
        """
        return dirichlet_expectation(self.pi, k)

    def natural_params(self, k):
        """
        Dirichlet natural parameters
        \alpha_{k} - 1
        """
        return self.pi[k] - 1
