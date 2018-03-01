import gmm_tools as gmm

num_samples = None
num_components = None
dimensions = None
cov_type = None

def _mean(l):
    return sum(l)/float(len(l))

def experiment_one(dimensions, num_components, num_samples, dt, cov_type, f):
    num_samples_list = list()
    accuracy_list = list()
    elapsed_time_list = list()
    num_iterations_list = list()
    lower_bound_list = list()

    while (num_samples <= 10000):
        num_samples_list.append(num_samples)
        acc_tmp = list()
        e_time_tmp = list()
        num_it_tmp = list()
        lb_tmp = list()
        for i in range(0,10):
            accuracy, elapsed_time, num_iterations, lower_bound = gmm.generate_and_fit_gmm_data(num_samples,
                                                                                                num_components,
                                                                                                num_components,
                                                                                                dimensions,
                                                                                                cov_type,
                                                                                                i)
            acc_tmp.append(accuracy)
            e_time_tmp.append(elapsed_time)
            num_it_tmp.append(num_iterations)
            lb_tmp.append(lower_bound)

        accuracy_list.append(_mean(acc_tmp))
        elapsed_time_list.append(_mean(e_time_tmp))
        num_iterations_list.append(_mean(num_it_tmp))
        lower_bound_list.append(_mean(lb_tmp))
        num_samples += dt

    f.write("\tnum_samples_list = " + str(num_samples_list) + "\n\n")
    f.write("\taccuracy_list = " + str(accuracy_list) + "\n\n")
    f.write("\tnum_iterations_list = " + str(num_iterations_list) + "\n\n")
    f.write("\telapsed_time_list = " + str(elapsed_time_list) + "\n\n")
    f.write("\tlower_bound_list = " + str(lower_bound_list) + "\n\n")

def experiment_two(dimensions, num_components, num_samples, dt, cov_type, f):
    dimensions_list = list()
    accuracy_list = list()
    elapsed_time_list = list()
    num_iterations_list = list()
    lower_bound_list = list()

    while (dimensions <= 50):
        dimensions_list.append(dimensions)
        acc_tmp = list()
        e_time_tmp = list()
        num_it_tmp = list()
        lb_tmp = list()
        for i in range(0,10):
            accuracy, elapsed_time, num_iterations, lower_bound = gmm.generate_and_fit_gmm_data(num_samples,
                                                                                                num_components,
                                                                                                num_components,
                                                                                                dimensions,
                                                                                                cov_type,
                                                                                                i)
            acc_tmp.append(accuracy)
            e_time_tmp.append(elapsed_time)
            num_it_tmp.append(num_iterations)
            lb_tmp.append(lower_bound)

        accuracy_list.append(_mean(acc_tmp))
        elapsed_time_list.append(_mean(e_time_tmp))
        num_iterations_list.append(_mean(num_it_tmp))
        lower_bound_list.append(_mean(lb_tmp))
        dimensions += dt

    f.write("\tdimensions_list = " + str(dimensions_list) + "\n\n")
    f.write("\taccuracy_list = " + str(accuracy_list) + "\n\n")
    f.write("\tnum_iterations_list = " + str(num_iterations_list) + "\n\n")
    f.write("\telapsed_time_list = " + str(elapsed_time_list) + "\n\n")
    f.write("\tlower_bound_list = " + str(lower_bound_list) + "\n\n")

def experiment_three(dimensions, num_components, num_samples, dt, cov_type, f):
    num_components_list = list()
    accuracy_list = list()
    elapsed_time_list = list()
    num_iterations_list = list()
    lower_bound_list = list()

    while (num_components <= 1005):
        num_components_list.append(num_components)
        acc_tmp = list()
        e_time_tmp = list()
        num_it_tmp = list()
        lb_tmp = list()
        for i in range(0,10):
            accuracy, elapsed_time, num_iterations, lower_bound = gmm.generate_and_fit_gmm_data(num_samples,
                                                                                                num_components,
                                                                                                num_components,
                                                                                                dimensions,
                                                                                                cov_type,
                                                                                                i)
            acc_tmp.append(accuracy)
            e_time_tmp.append(elapsed_time)
            num_it_tmp.append(num_iterations)
            lb_tmp.append(lower_bound)

        accuracy_list.append(_mean(acc_tmp))
        elapsed_time_list.append(_mean(e_time_tmp))
        num_iterations_list.append(_mean(num_it_tmp))
        lower_bound_list.append(_mean(lb_tmp))
        num_components += dt

    f.write("\tnum_components_list = " + str(num_components_list) + "\n\n")
    f.write("\taccuracy_list = " + str(accuracy_list) + "\n\n")
    f.write("\tnum_iterations_list = " + str(num_iterations_list) + "\n\n")
    f.write("\telapsed_time_list = " + str(elapsed_time_list) + "\n\n")
    f.write("\tlower_bound_list = " + str(lower_bound_list) + "\n\n")

print("STARTING EXPERIMENT ONE!")
# EXPERIMENT ONE: SAMPLE SIZE
# * dimensionality = 5
# * number of components = 100
# * sample size = (500, 10000), dt = 500

f = open('experiment_one.log', 'w')
dimensions = 5
num_components = 100
dt = 500

f.write("TESTING SAMPLE SIZE")
f.write("\tdimensions = %d\n" % dimensions)
f.write("\tnum_components = %d\n" % num_components)
f.write("\tnum_samples = (500,10000)\n")
f.write("\tdt = %d\n\n" % dt)

### full
cov_type = "full"
num_samples = 500

f.write("TESTING COVARIANCE TYPE: %s\n" % cov_type)
experiment_one(dimensions, num_components, num_samples, dt, cov_type, f)

### diag
cov_type = "diag"
num_samples = 500

f.write("TESTING COVARIANCE TYPE: %s\n" % cov_type)
experiment_one(dimensions, num_components, num_samples, dt, cov_type, f)

### spherical
cov_type = "spherical"
num_samples = 500

f.write("TESTING COVARIANCE TYPE: %s\n" % cov_type)
experiment_one(dimensions, num_components, num_samples, dt, cov_type, f)

### tied
cov_type = "tied"
num_samples = 500

f.write("TESTING COVARIANCE TYPE: %s\n" % cov_type)
experiment_one(dimensions, num_components, num_samples, dt, cov_type, f)

f.close()

print("STARTING EXPERIMENT TWO!")
# Experiment Two: Dimensionality
# * dimensionality = (2, 50), dt = 4
# * number of components = 100
# * sample size = 10000

f = open('experiment_two.log', 'w')
num_components = 100
num_samples = 10000
dt = 4

### full
cov_type = "full"
dimensions = 2

f.write("TESTING COVARIANCE TYPE: %s\n" % cov_type)
experiment_two(dimensions, num_components, num_samples, dt, cov_type, f)

### diag
cov_type = "diag"
dimensions = 2

f.write("TESTING COVARIANCE TYPE: %s\n" % cov_type)
experiment_two(dimensions, num_components, num_samples, dt, cov_type, f)

### spherical
cov_type = "spherical"
dimensions = 2

f.write("TESTING COVARIANCE TYPE: %s\n" % cov_type)
experiment_two(dimensions, num_components, num_samples, dt, cov_type, f)

### tied
cov_type = "tied"
dimensions = 2

f.write("TESTING COVARIANCE TYPE: %s\n" % cov_type)
experiment_two(dimensions, num_components, num_samples, dt, cov_type, f)

f.close()

print("STARTING EXPERIMENT THREE!")
# Experiment Three: Number of Components
# * dimensionality = 20
# * number of components = (5, 1005), dt = 100
# * sample size = 10000

f = open('experiment_three.log', 'w')
dimensions = 20
num_samples = 10000
dt = 100

### full
cov_type = "full"
num_components = 5

f.write("TESTING COVARIANCE TYPE: %s\n" % cov_type)
experiment_three(dimensions, num_components, num_samples, dt, cov_type, f)

### diag
cov_type = "diag"
num_components = 5

f.write("TESTING COVARIANCE TYPE: %s\n" % cov_type)
experiment_three(dimensions, num_components, num_samples, dt, cov_type, f)

### spherical
cov_type = "spherical"
num_components = 5

f.write("TESTING COVARIANCE TYPE: %s\n" % cov_type)
experiment_three(dimensions, num_components, num_samples, dt, cov_type, f)

### tied
cov_type = "tied"
num_components = 5

f.write("TESTING COVARIANCE TYPE: %s\n" % cov_type)
experiment_three(dimensions, num_components, num_samples, dt, cov_type, f)

f.close()

