import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat


def _generate_covariances(components, dimensions, diagonal=False, isotropic=False):
    """Generates a batch of random positive definite covariance matrices"""
    covariances = np.zeros((components, dimensions, dimensions))

    if isotropic:
        for i in range(components):
            covariances[i] = np.diag(np.full((dimensions,), np.abs(np.random.normal())))
    elif diagonal:
        for i in range(components):
            covariances[i] = np.diag(np.abs(np.random.normal(size=[dimensions])))
    else:
        for i in range(components):
            covariances[i] = np.random.normal(size=[dimensions, dimensions])
            covariances[i] = np.dot(covariances[i], covariances[i].T)

    return covariances


def generate_gmm_data(size, components, dimensions, seed=None, diagonal=False, isotropic=False):
    """Generates synthetic data of a given size from a random Gaussian Mixture Model"""
    np.random.seed(seed)

    means = np.random.normal(size=[components, dimensions]) * 10
    covariances = _generate_covariances(components, dimensions, diagonal, isotropic)
    weights = np.abs(np.random.normal(size=[components]))
    weights /= np.sum(weights)

    result = np.empty((size, dimensions), dtype=np.float64)
    responsibilities = np.empty((size,), dtype=np.int32)

    for i in range(size):
        comp = np.random.choice(components, p=weights)

        responsibilities[i] = comp
        result[i] = np.random.multivariate_normal(
            means[comp], covariances[comp]
        )

    np.random.seed()

    return result, means, covariances, weights, responsibilities


def generate_cmm_data(size, components, dimensions, seed=None, count_range=(2, 100)):
    """Generates synthetic data of a given size from a random Categorical Mixture Model"""
    np.random.seed(seed)

    counts = np.random.randint(
        count_range[0], count_range[1],
        (dimensions,)
    )

    means = []
    for comp in range(components):
        comp_means = []
        for dim in range(dimensions):
            comp_means.append(np.random.uniform(0.25, 0.75, (counts[dim],)))
            comp_means[-1] /= np.sum(comp_means[-1])
        means.append(comp_means)

    weights = np.abs(np.random.normal(size=[components]))
    weights /= np.sum(weights)

    result = np.empty((size, dimensions), dtype=np.int32)
    responsibilities = np.empty((size,), dtype=np.int32)

    for i in range(size):
        comp = np.random.choice(components, p=weights)

        responsibilities[i] = comp
        for dim in range(dimensions):
            result[i, dim] = np.random.choice(
                counts[dim], p=means[comp][dim]
            )

    np.random.seed()

    return result, counts, means, weights, responsibilities


def generate_cgmm_data(size, components, c_dimensions, g_dimensions, seed=None,
                       count_range=(2, 100), diagonal=False, isotropic=False):
    """Generates synthetic data of a given size from a random Categorical + Gaussian Mixture Model"""
    np.random.seed(seed)

    c_counts = np.random.randint(
        count_range[0], count_range[1],
        (c_dimensions,)
    )

    c_means = []
    for comp in range(components):
        comp_c_means = []
        for dim in range(c_dimensions):
            comp_c_means.append(np.random.uniform(0.25, 0.75, (c_counts[dim],)))
            comp_c_means[-1] /= np.sum(comp_c_means[-1])
        c_means.append(comp_c_means)

    g_means = np.random.normal(size=[components, g_dimensions]) * 10
    g_covariances = _generate_covariances(components, g_dimensions, diagonal, isotropic)

    weights = np.abs(np.random.normal(size=[components]))
    weights /= np.sum(weights)

    c_result = np.empty((size, c_dimensions), dtype=np.int32)
    g_result = np.empty((size, g_dimensions), dtype=np.float64)
    responsibilities = np.empty((size,), dtype=np.int32)

    for i in range(size):
        comp = np.random.choice(components, p=weights)
        responsibilities[i] = comp

        for dim in range(c_dimensions):
            c_result[i, dim] = np.random.choice(
                c_counts[dim], p=c_means[comp][dim]
            )

        g_result[i] = np.random.multivariate_normal(
            g_means[comp], g_covariances[comp]
        )

    np.random.seed()

    return c_result, g_result, c_counts, c_means, g_means, g_covariances, weights, responsibilities


def _plot_gaussian(mean, covariance, color, zorder=0):
    """Plots the mean and 2-std ellipse of a given Gaussian"""
    plt.plot(mean[0], mean[1], color[0] + ".", zorder=zorder)

    if covariance.ndim == 1:
        covariance = np.diag(covariance)

    radius = np.sqrt(5.991)
    eigvals, eigvecs = np.linalg.eig(covariance)
    axis = np.sqrt(eigvals) * radius
    slope = eigvecs[1][0] / eigvecs[1][1]
    angle = 180.0 * np.arctan(slope) / np.pi

    plt.axes().add_artist(pat.Ellipse(
        mean, 2 * axis[0], 2 * axis[1], angle=angle,
        fill=False, color=color, linewidth=1, zorder=zorder
    ))


def plot_fitted_data(data, means, covariances, true_means=None, true_covariances=None):
    """Plots the data and given Gaussian components"""
    plt.plot(data[:, 0], data[:, 1], "b.", markersize=0.5, zorder=0)

    if true_means is not None:
        for i in range(len(true_means)):
            _plot_gaussian(true_means[i], true_covariances[i], "green", 1)

    for i in range(len(means)):
        _plot_gaussian(means[i], covariances[i], "red", 2)

    plt.show()
