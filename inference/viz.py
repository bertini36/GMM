# -*- coding: UTF-8 -*-

from __future__ import absolute_import

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


def create_cov_ellipse(cov, pos, nstd=2, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    return ellip


def plot_iteration(ax_spatial, circs, sctZ, lambda_m, cov, xn, n_iters, K):
    """
    Plot the Gaussians in every iteration
    """
    if n_iters == 0:
        plt.scatter(xn[:, 0], xn[:, 1], cmap=cm.gist_rainbow, s=5)
        sctZ = plt.scatter(lambda_m[:, 0], lambda_m[:, 1],
                           color='black', s=5)
    else:
        for circ in circs: circ.remove()
        circs = []
        for k in range(K):
            if type(cov) is list:
                circ = create_cov_ellipse(cov[k], lambda_m[k, :],
                                          color='r', alpha=0.3)
            else:
                circ = create_cov_ellipse(cov, lambda_m[k, :],
                                          color='r', alpha=0.3)
            circs.append(circ)
            ax_spatial.add_artist(circ)
        sctZ.set_offsets(lambda_m)
    plt.draw()
    plt.pause(0.001)
    return ax_spatial, circs, sctZ
