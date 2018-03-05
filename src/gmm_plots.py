import gmm_tools as gmm

num_samples = 10000
num_components = 5
dimensions = 5

gmm.generate_and_fit_gmm_data(num_samples,
                              num_components,
                              num_components,
                              dimensions,
                              "full",
                              0,
                              True,
                              "Full Covariance")
gmm.generate_and_fit_gmm_data(num_samples,
                              num_components,
                              num_components,
                              dimensions,
                              "diag",
                              0,
                              True,
                              "Diagonal Covariance")
gmm.generate_and_fit_gmm_data(num_samples,
                              num_components,
                              num_components,
                              dimensions,
                              "spherical",
                              0,
                              True,
                              "Spherical Covariance")
gmm.generate_and_fit_gmm_data(num_samples,
                              num_components,
                              num_components,
                              dimensions,
                              "tied",
                              0,
                              True,
                              "Tied Covariance")
