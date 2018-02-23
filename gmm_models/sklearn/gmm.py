import itertools
import math

import sklearn
import numpy as np
from scipy import linalg

import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as mpl

import plotly
import plotly.plotly as py
import plotly.graph_objs as go

from sklearn import mixture

# color_iter = itertools.cycle(['navy', 'cornflowerblue', 'gold',
#                               'darkorange'])

color_iter = itertools.cycle(['navy', 'cornflowerblue', 'gold',
                              'darkorange', 'black', 'burlywood',
                              'darkmagenta', 'darkgoldenrod',
                              'lavender', 'lawngreen', 'hotpink',
                              'honeydew', 'indianred', 'indigo',
                              'maroon', 'olive', 'peru', 'plum',
                              'salmon', 'seashell', 'sienna', 'silver',
                              'steelblue', 'thistle', 'tomato'])

# plotly_results(data, gmm.predict(data), gmm.means_, gmm.covariances_,'Gaussian Mixture')
def plotly_results(X, Y_, means, covariances, cov_type, title):
    data = []
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
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

        elle = go.Scatter(x=x_ , y=y_, mode='lines',
                          showlegend=False, line=dict(color='black',
                                    width=2))
        data.append(elle)

    layout = go.Layout(title=title, showlegend=False,
                       xaxis=dict(zeroline=False, showgrid=False),
                       yaxis=dict(scaleanchor = "x",zeroline=False, showgrid=False),)
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig)

def generate_gmm_data(points, components, dimensions, seed):
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

def plot_fitted_data(points, c_means, c_variances):
    """Plots the data and given Gaussian components"""
    plt.plot(points[:, 0], points[:, 1], "b.", zorder=0)
    plt.plot(c_means[:, 0], c_means[:, 1], "r.", zorder=1)
    print(c_means)
    print(c_variances)

    for i in range(c_means.shape[0]):
        std = np.sqrt(c_variances[i])
        plt.axes().add_artist(pat.Ellipse(
            c_means[i], 2 * std[0], 2 * std[1],
            fill=False, color="red", linewidth=2, zorder=1
        ))
    plt.show()

def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

        # plt.xlim(-9., 5.)
        # plt.ylim(-3., 6.)
        plt.xticks(())
        plt.yticks(())
        plt.title(title)


# Number of samples per component
n_samples = 5
gen_components = 2
fit_components = 2
dimensions = 2
cov_type = 'spherical'
seed = 0
# C = np.array([[0., -0.1], [1.7, .4]])
# X = np.r_[np.dot(np.random.randn(n_samples, components), C),
#           .7 * np.random.randn(n_samples, components) + np.array([-6, 3])]

data, true_means, true_variances, true_weights = generate_gmm_data(n_samples, gen_components, dimensions, seed)
print(data)

data2, true_means, true_variances, true_weights = generate_gmm_data(n_samples, gen_components, dimensions, seed)
print(data2)

# Fit a Gaussian mixture with EM using five components
# gmm = mixture.GaussianMixture(n_components=fit_components, covariance_type=cov_type, tol=1e-06, verbose=3, verbose_interval=1).fit(data)
# plotly_results(data, gmm.predict(data), gmm.means_, gmm.covariances_, cov_type, 'Gaussian Mixture')
# plotly_results(data, gmm)

# plot_fitted_data(data, gmm.means_, gmm.covariances_)

# Fit a Dirichlet process Gaussian mixture using five components
# dpgmm = mixture.BayesianGaussianMixture(n_components=5,
#                                         covariance_type='full').fit(X)
# plot_results(X, dpgmm.predict(X), dpgmm.means_, dpgmm.covariances_, 1,
#              'Bayesian Gaussian Mixture with a Dirichlet process prior')

# plt.show()
