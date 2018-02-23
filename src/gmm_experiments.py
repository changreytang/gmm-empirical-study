import gmm_tools as gmm

NUM_SAMPLES = 500
NUM_COMPONENTS_GENERATED = 2
NUM_COMPONENTS_FITTED = 2
DIMENSIONS = 2
COV_TYPE = 'diag'
SEED = 0

gmm.generate_fit_and_plot_gmm_data(NUM_SAMPLES,
                                   NUM_COMPONENTS_GENERATED,
                                   NUM_COMPONENTS_FITTED,
                                   DIMENSIONS,
                                   COV_TYPE,
                                   SEED)


