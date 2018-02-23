import itertools
import math
import numpy as np
from scipy import linalg
import sklearn
from sklearn import mixture

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as pat

import plotly
import plotly.plotly as py
import plotly.graph_objs as go

COLOR_ITER = itertools.cycle(['navy', 'cornflowerblue', 'gold',
                              'darkorange', 'black', 'burlywood',
                              'darkmagenta', 'darkgoldenrod',
                              'lavender', 'lawngreen', 'hotpink',
                              'honeydew', 'indianred', 'indigo',
                              'maroon', 'olive', 'peru', 'plum',
                              'salmon', 'seashell', 'sienna', 'silver',
                              'steelblue', 'thistle', 'tomato'])

def _plot_results(X, Y_, means, covariances, cov_type, title):
    """Plots the ellipses of the GMM using plotly"""
    data = []
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, COLOR_ITER)):
        if cov_type == 'diag':
            covar = np.diag(covar[:2])
        elif cov_type == 'spherical':
            covar = np.eye(means.shape[1])*covar
        elif cov_type == 'tied':
            covar = covariances

        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        trace = go.Scatter(x=X[Y_ == i, 0], y=X[Y_ == i, 1],
                           mode='markers',
                           marker=dict(color=color))
        data.append(trace)
        # Plot an ellipse to show the Gaussian component
        a =  v[1]
        b =  v[0]
        x_origin = mean[0]
        y_origin = mean[1]
        x_ = [ ]
        y_ = [ ]

        for t in range(0,361,10):
            x = a*(math.cos(math.radians(t))) + x_origin
            x_.append(x)
            y = b*(math.sin(math.radians(t))) + y_origin
            y_.append(y)

        elle = go.Scatter(x=x_,
                          y=y_,
                          mode='lines',
                          showlegend=False,
                          line=dict(color='black',
                                    width=2))

        data.append(elle)

    layout = go.Layout(title=title, showlegend=False,
                       xaxis=dict(zeroline=False, showgrid=False),
                       yaxis=dict(scaleanchor = "x",zeroline=False, showgrid=False),)
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig)

def _generate_gmm_data(points, components, dimensions, seed):
    """Generates synthetic data of a given size from a random GMM"""
    np.random.seed(seed)

    c_means = np.random.normal(size=[components, dimensions]) * 10
    c_variances = np.abs(np.random.normal(size=[components, dimensions]))
    c_weights = np.abs(np.random.normal(size=[components]))
    c_weights /= np.sum(c_weights)

    result = np.zeros((points, dimensions), dtype=np.float32)

    for i in range(points):
        comp = np.random.choice(np.array(range(components)), p=c_weights)
        result[i] = np.random.multivariate_normal(
            c_means[comp], np.diag(c_variances[comp])
        )
    np.random.seed()
    return result, c_means, c_variances, c_weights

def generate_fit_and_plot_gmm_data(num_samples, num_comp_gen, num_comp_fit, dim, cov_type, seed):
    data, true_means, true_variances, true_weights = _generate_gmm_data(num_samples,
                                                                       num_comp_gen,
                                                                       dim,
                                                                       seed)
    gmm = mixture.GaussianMixture(n_components=num_comp_fit,
                                  covariance_type=cov_type,
                                  tol=1e-06,
                                  verbose=2,
                                  verbose_interval=1).fit(data)

    _plot_results(data, gmm.predict(data), gmm.means_, gmm.covariances_, cov_type, 'Gaussian Mixture')

