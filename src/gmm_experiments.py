import gmm_tools as gmm

num_samples = None
num_components = None
dimensions = None
cov_type = None
seed = 0

# EXPERIMENT ONE: SAMPLE SIZE
# * dimensionality = 5
# * number of components = 100
# * sample size = (500, 10000), dt = 500

###


# Experiment Two: Dimensionality
# * dimensionality = (2, 50), dt = 4
# * number of components = 100
# * sample size = 10000

# Experiment Three: Number of Components
# * dimensionality = 20
# * number of components = (5, 1005), dt = 100
# * sample size = 10000

NUM_SAMPLES = 10000
NUM_COMPONENTS_GENERATED = 5005
NUM_COMPONENTS_FITTED = 5005
DIMENSIONS = 5
COV_TYPE = 'tied'
SEED = 0

accuracy, elapsed_time, num_iterations, lower_bound = gmm.generate_fit_and_plot_gmm_data(NUM_SAMPLES,
                                                                                         NUM_COMPONENTS_GENERATED,
                                                                                         NUM_COMPONENTS_FITTED,
                                                                                         DIMENSIONS,
                                                                                         COV_TYPE,
                                                                                         SEED)


gmm.print_data(NUM_SAMPLES, NUM_COMPONENTS_GENERATED, NUM_COMPONENTS_FITTED, DIMENSIONS, COV_TYPE, SEED, accuracy, elapsed_time, num_iterations, lower_bound)

