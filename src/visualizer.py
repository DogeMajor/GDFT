from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from utils import extract_thetas_records, seq_norm, approximate_matrix, approximate_phases
from gdft import dft_matrix, gdft_matrix, two_param_gdft_matrix
from analyzer import ThetasAnalyzer
from correlations import CorrelationAnalyzer, Correlation
from derivator import *

plt.grid(True)


def orderings_dist(thetas):
    orderings = np.argsort(thetas)
    natural_order = list(range(thetas.shape[0]))
    return seq_norm(natural_order, orderings) / thetas.shape[0]


def find_best_orderings(thetas_collections):
    records = [(theta, corr) for theta, corr in zip(thetas_collections.thetas, theta_collections.correlations)]
    def dist(item):
        return orderings_dist(item[0])
    return sorted(records, key=dist)


def complex_to_coords(thetas):
    length = np.absolute(thetas)
    return length*np.cos(thetas), length*np.sin(thetas)


def to_coords(thetas):
    return np.cos(thetas), np.sin(thetas)


def rotate_to_center(thetas, deg_angle):
    angle = deg_angle * (np.pi / 180)
    return thetas - angle


def generate_points(thetas):
    length = thetas.shape[0]
    args = np.array(list(range(length)))
    return args, thetas


def angle_dist(theta_collection, partition=None):
    if partition is None:
        partition = theta_collection[0].shape[0]
    dist = {key: 0 for key in range(partition+1)}
    divider = np.pi/partition
    print("dist", dist, divider)
    for theta_vecs in theta_collection:
        for theta in theta_vecs:
            key = int(theta / divider)
            dist[key] += 1
    return dist


def angle_probabilities(theta_collection, partition=None):
    angle_distr = angle_dist(theta_collection, partition)
    length = sum(angle_distr.values())
    return {key: value / length for key, value in angle_distr.items()}


def plot_eigenvalues(theta):
    gdft = gdft_matrix(theta.shape[0], theta)
    eig_vals = np.linalg.eig(gdft)[0]
    plt.plot(eig_vals.real, eig_vals.imag, 'x')


def fit_polynome(thetas, grade):
    args, thetas = generate_points(thetas)
    z = np.polyfit(args, thetas, grade)
    f = np.poly1d(z)
    return f


def plot_fitted_polynome(pol_fn, thetas):
    args, thetas = generate_points(thetas)
    x_new = np.linspace(args[0], args[-1], 50)
    y_new = pol_fn(x_new)
    print("ynew", type(y_new))
    plt.plot(args, thetas, 'o', x_new, y_new)


def plot_angles(thetas):
    args, thetas = generate_points(thetas)
    plt.plot(args, thetas, 'x')


def polar_plot_angles(thetas):
    x, y = to_coords(thetas)
    plt.plot(x, y, 'o')


def polar_plot_numbered_angles(thetas):
    x_coords, y_coords = to_coords(thetas)
    coords = list(zip(x_coords, y_coords))
    for index, (x, y) in enumerate(coords):
        plt.plot(x, y, 'o')
        plt.text(0.9*x, 0.9*y, str(index))


def plot_polynome_roots(polynome, max_root_len=None):
    if max_root_len is None:
        max_root_len = len(polynome.c)
    coeffs = polynome.r
    if max(np.abs(coeffs)) < max_root_len:
        x, y = coeffs.real, coeffs.imag
        plt.plot(x, y, 'o')


def plot_fn(fn, dim):
    x_new = np.linspace(0, dim-1, 50)
    y_new = fn(x_new)
    plt.plot(x_new, y_new)


if __name__ == "__main__":
    thetas_analyzer = ThetasAnalyzer(8)
    #theta_collections = extract_thetas_records("../data/", "10thetas_16x16__12-27_15_38.json")
    #theta_collections = extract_thetas_records("../data/", "30thetas_16x16__1-1_21_14.json")
    #theta_collections = extract_thetas_records("../data/", "10thetas_16x16__12-27_11_58.json")
    #theta_collections = thetas = extract_thetas_records("../data/", "10thetas_16x16__12-26_19_4.json")
    #theta_collections = thetas = extract_thetas_records("../data/", "100thetas_4x4__12-26_16_6.json")

    #theta_collections = extract_thetas_records("../data/", "R_ac_100thetas_8x8__3-7_14_39.json")
    #theta_collections = extract_thetas_records("../data/", "d_ac_100thetas_8x8__3-7_15_10.json")
    theta_collections = extract_thetas_records("../data/", "R_ac_30thetas_8x8__3-10_13_3.json")

    #theta_collections = extract_thetas_records("../data/", "results_2018-12-24 23_33.json")
    
    sorted_thetas = thetas_analyzer.sort_thetas(theta_collections.thetas, 6)
    #thetas0 = sorted_thetas.thetas[0]
    #data_matrix = thetas_analyzer.to_data_matrix(theta_collections.thetas)
    #reduced_cov_mats = thetas_analyzer.cov_pca_reductions(sorted_thetas, cutoff_ratio=0.05)
    #print(len(reduced_cov_mats))
    #print(reduced_cov_mats)

    '''for theta in thetas0:
        polar_plot_angles(theta)
        print(theta)

    plt.show()'''
    #theta = np.array([0.43641541, 1.21234458, 3.14159265, 2.98732434, 2.05067131, 0.33163315,
    #                  2.81226277, 0.1630281])

    def symm_checker(theta, mu, m=0):
        N = theta.shape[0]

        def get_component(sigma):
            gets_addition = sigma + mu < N
            gets_subtraction = sigma - mu < N
            lhs = 0
            if gets_addition and gets_subtraction:
                lhs += 0.5*theta[sigma + mu]
                lhs += 0.5 * theta[sigma - mu]
            return theta[sigma] - lhs

        symmetries = [get_component(index) for index, comp in enumerate(theta)]
        return symmetries



    #for n in range(8):
    #    print(symm_checker(theta, n))
    #
    #theta8 = np.array([0.4364, 1.2123, 1.9882, 2.7641, 2.5646, 2.3651, 2.1656, 1.9661])
    theta8 = theta_collections.thetas[1]
    #print(theta8)
    '''gdft = gdft_matrix(8, theta8)
    corrs = thetas_analyzer.get_correlations(gdft)
    print(corrs)
    sigma = 0
    alpha = beta = 0
    mu = 1
    print(derivative(gdft, sigma, alpha, beta, mu))
    print(gradient(gdft, alpha, beta, mu))'''


    corr_analyzer = CorrelationAnalyzer(8)
    #diffs = np.array([difference(corr_analyzer, theta8, ind, "avg_auto_corr", h=0.00001) for ind in range(8)])
    #print(diffs)
    #num_grad = grad(corr_analyzer, theta8, "max_auto_corr", h=0.00001)
    #print("max_auto_corr", num_grad)
    #print("avg_auto_corr", grad(corr_analyzer, theta8, "avg_auto_corr", step=0.00001))
    #[-6.55475674e-07+0.j -9.05941988e-07+0.j -1.09423581e-06+0.j -1.21858079e-06+0.j -1.21858079e-06+0.j -1.09423581e-06+0.j
    #-9.05941988e-07+0.j -6.55475674e-07+0.j]
    #theta3 = np.zeros(8)
    #theta3[0:4] = np.arange(0, 4, 1)
    #print(theta3)
    #print("avg_auto_corr", grad(corr_analyzer, theta3, "avg_auto_corr", step=0.0001))

    #diff_ct_0 = diff_c_tensor(theta8, 0, h=0.001)
    #print(diff_ct_0)

    #theta2 = theta_collections.thetas[10]
    #print("avg_auto_corr", grad(corr_analyzer, theta2, "max_auto_corr", step=0.0001))
    #diffs = [difference(corr_analyzer, theta8, index, "avg_auto_corr", h=0.001) for index in range(8)]
    #print(diffs)
    #print("max_cross_corr", grad(corr_analyzer, theta8, "max_cross_corr", h=0.00001))
    #print("avg_cross_corr", grad(corr_analyzer, theta8, "avg_cross_corr", h=0.00001))

    direction = np.array([-2, -1, 0, 1,
                          2, 3, 4, 5])
    theta_ref2 = np.array([0.23263316, 1.06778065, 3.05624654, 2.96119473,
                           2.08375977, 0.4239405, 2.96378942, 0.37377238])
    direction2 = np.array([-7, -5, -3, -1,
                           1, 3, 5, 7])
    #print(symm_checker(new_theta, 1))
    '''theta_ref = sorted_thetas.thetas[0][0]
    print(theta_ref)
    for theta in sorted_thetas.thetas[0][1:]:
        gdft = gdft_matrix(8, theta)
        avg_auto_corr_gradient = [auto_corr_derivative(sigma, gdft) for sigma in range(8)]
        #print(avg_auto_corr_gradient)
        diff = theta_ref - theta
        maximum = np.min(np.abs(diff))
        print(diff/maximum)'''
    delta = 2*np.pi/200
    #Correlations(max_auto_corr=0.125, avg_auto_corr=(0.0857138932287049+0j), max_cross_corr=0.6430321034974654, avg_cross_corr=(0.9877551581101849+0j)
    theta0 = np.array([0.1574527, 2.77965783, 0.27200508, 1.96402313,
                       2.87364471, 3.00086244, 1.04457988, 0.24164262])
    constr_theta6 = [0.1574527, 2.77965783, 0.27200508, 1.96402313,
                     np.pi - 0.27200508, np.pi - 0.1574527, 1.04457988, 0.27200508]
    #direction = np.array([0, -1, 0, -1,
    #                      0, -1, 0, -1])



    constructed_theta = [0.27200508, np.pi - 0.27200508, 0.27200508, 1.96402313,
                         np.pi - 0.27200508, 3.00086244, 3.00086244-1.96402313, np.pi - 3.00086244]

    theta_init = np.array([0.40135885, 1.19480564, 3.14158074, 3.0048373,
                           2.0857035, 0.38419745, 2.88234682, 0.25062983])

    opt_theta = np.array([0.40135885, np.pi/2 - 0.40135885, 2.88234682+0.25062983, np.pi+0.25062983-0.40135885,
                          2.0857035, 0.40135885, np.pi-0.25062983, 0.25062983])
    #print(theta0+ np.pi*direction)
    '''for n in range(-4, 4):

        constr_gdft = gdft_matrix(8, constructed_theta+n*delta*direction)
        correlations = corr_analyzer.get_correlations(constr_gdft)
        print(n, correlations.avg_auto_corr)
        print("R_ac gradient length")
        print(np.linalg.norm(auto_corr_gradient(constr_gdft)))'''
    '''for n in range(0, 5):
        new_theta = theta_init - n*0.015*direction
        constr_gdft = gdft_matrix(8, new_theta)
        correlations = corr_analyzer.get_correlations(constr_gdft)
        print(n, correlations)'''
        #print(new_theta)
    #polar_plot_angles(new_theta)
    example_theta = np.array([0.43171271, 1.20998793, 3.14159265, 2.9896899,
                              2.05536774, 0.33869695, 2.82167874, 0.17477883])
    #print(theta_init - example_theta)
    #print(theta_init + 10*0.015*direction)
    #for theta in sorted_thetas.thetas[0][0:10]:
    #    print(theta)
    #polar_plot_numbered_angles(new_theta)
    #plt.show()

    '''c_analyzer3 = CorrelationAnalyzer(3)
    for n in range(200):
        thetas3 = np.array([2+delta*n, 0, 1-delta*n])

        gdft3 = gdft_matrix(3, thetas3)
        correlations = c_analyzer3.get_correlations(gdft3)
        print(correlations.avg_auto_corr)'''

    '''
    poly_coeff_8 = np.array([-7.47998864e-03, 1.73916258e-01, -1.61020449e+00, 7.60456544e+00,
                             -1.93127379e+01, 2.45158151e+01, -1.05428434e+01, 2.47251476e-01])

    poly_coeff_4 = np.array([1.04719702, -4.05332898, 2.84656905, 2.63384441])
   
    #THETAS8 = np.array([-3.12, -3.38, -3.2, -1.86, -1.27, 0.06, 0.25, -0.01])
    THETAS8 = np.array([1.637, -0.79, -0.54, 2.01, 1.59, -0.83, 1.73, 2.44])

    #gdft = gdft_matrix(8, THETAS8)
    gdft = two_param_gdft_matrix(8, np.zeros(8), THETAS8)
    correlations = thetas_analyzer.get_correlations(dft_matrix(8))
    print(correlations)
    '''
    #Correlations(max_auto_corr=0.3814517374888245, avg_auto_corr=(1.0209859493694915+0j),
    #max_cross_corr=0.5286804616397811, avg_cross_corr=(0.854144864375787+0j), avg_merit_factor=(0.9794454082522375+0j))'''