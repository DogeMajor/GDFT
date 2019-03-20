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
    thetas_analyzer = ThetasAnalyzer(16)
    #theta_collections = extract_thetas_records("../data/", "10thetas_16x16__12-27_15_38.json")
    #theta_collections = extract_thetas_records("../data/", "30thetas_16x16__1-1_21_14.json")
    #theta_collections = extract_thetas_records("../data/", "10thetas_16x16__12-27_11_58.json")
    #theta_collections = thetas = extract_thetas_records("../data/", "10thetas_16x16__12-26_19_4.json")
    #theta_collections = thetas = extract_thetas_records("../data/", "100thetas_4x4__12-26_16_6.json")

    #theta_collections = extract_thetas_records("../data/", "R_ac_100thetas_8x8__3-7_14_39.json")
    #theta_collections = extract_thetas_records("../data/", "d_ac_100thetas_8x8__3-7_15_10.json")
    #theta_collections = extract_thetas_records("../data/", "R_ac_30thetas_8x8__3-10_13_3.json")
    theta_collections = extract_thetas_records("../data/", "R_ac_30thetas_16x16__3-4_21_58.json")

    #theta_collections = extract_thetas_records("../data/", "results_2018-12-24 23_33.json")

    #print(theta_collections)
    #sorted_thetas = thetas_analyzer.sort_thetas(theta_collections.thetas, 6)
    #print(sorted_thetas.thetas[0])

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

    corr_analyzer = CorrelationAnalyzer(16)

    #The only 'symmetric' theta we found (due to lilitations theta_k in (0, np.pi) ^N
    special_theta8 = np.array([1.44608874, -0.24, 3.14159265, 2.35929684,
                              2.35929684, 3.14159265, -0.24, 1.44608874])

    direction0 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    direction1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

    u0 = direction0 / np.sqrt(direction0.dot(direction0))
    u1 = direction1 - direction1.dot(u0) * u0
    u1 = u1 / np.sqrt(u1.dot(u1))
    P = np.outer(u0, u0) + np.outer(u1, u1)
    #print(np.max(np.abs(P @ P - P)))
    #print(P @ np.array([1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16]))


    data_matrix = thetas_analyzer.to_data_matrix(theta_collections.thetas[0:10], subtract_avgs=True)

    #print(sorted_thetas.thetas[0])

    thetas_group0 = [np.array([3.14159265, 0., 0.19933716, 3.02679522,
                               0.38564941, 1.09367971, 2.74774895, 2.52162586,
                               3.14159265, 0., 3.14159265, 1.74309246,
                               2.83619393, 1.10752226, 0., 0.]),
                     np.array([3.14159265, 0.        , 0.        , 3.14159265, 3.14159265,
                               0.        , 3.14159265, 1.20035367, 2.87736858, 0.85437013,
                               0.33549079, 0.04950922, 0.69988427, 2.61669993, 1.75469542, 2.64590917]),
                     np.array([3.09925284, 3.14159265, 0.11008461, 0.        , 0.        ,
                               3.14159265, 3.14159265, 0.        , 2.88918522, 2.85219038,
                               0.45492782, 2.68947348, 1.36337871, 2.59114784, 1.02699643, 2.97297499]),
                     np.array([3.09064294, 3.14159265, 0.46907206, 2.45890132, 3.12389768,
                               0.74730237, 0.42638899, 0.21635776, 1.59405698, 3.00169057,
                               3.14159265, 2.37156865, 0.        , 3.14159265, 0.        , 3.14159265]),
                     np.array([3.14159265, 0.        , 0.        , 3.03454093, 0.04242189,
                               0.07334303, 3.13327875, 0.98784566, 2.39633759, 0.7076594 ,
                               3.07091944, 2.82186889, 3.14159265, 0.74341578, 0.59469586, 0.88990693])]
    theta_avg0 = sum(thetas_group0) / len(thetas_group0)
    print("group_avg:")
    print(theta_avg0)
    from sequencefinder import SequenceFinder
    seq_finder = SequenceFinder()
    theta_diff01 = theta_collections.thetas[0] - theta_collections.thetas[1]
    vec01 = theta_diff01 - P @ theta_diff01

    theta_ref = theta_collections.thetas[1]
    cross_corr_grad = auto_corr_gradient(gdft_matrix(16, theta_ref))
    print("gradient")
    print(cross_corr_grad)
    '''for theta in theta_collections.thetas[0:0] + theta_collections.thetas[1:]:

        tot_theta_diff = theta - theta_ref
        # print(tot_theta_diff)
        theta_diff = tot_theta_diff - P @ tot_theta_diff
        #print(theta_diff)
        print(seq_finder._to_integers(theta_diff))
        theta_diff /= np.sqrt(tot_theta_diff.dot(tot_theta_diff))
        ratio = np.sqrt(theta_diff.dot(theta_diff)) / np.sqrt(tot_theta_diff.dot(tot_theta_diff))
        print(ratio)'''




    real_theta_ref = np.array([1.50433103, 3.2699411, 0.77086939, 2.73460937,
                               0.07997835, -0.02197093, 0.34883782, -0.26989642,
                               2.04157638, 2.22135669, 3.19397327, 0.78379947,
                               -0.05472408, 3.38375429, 2.88657516, -0.07628343])

    real_theta_ref2 = np.array([1.63726153, -0.12834981, 2.3707224, 0.40698071,
                                3.06162368, 3.16357145, 2.79276685, 3.41148958,
                                1.10002122, 0.92024312, -0.05238471, 2.3578042,
                                3.19632276, -0.24215659, 0.25502003, 3.21787861])

    old_gdft = gdft_matrix(16, real_theta_ref)
    special_direction = np.array([102, -161, 333, -62, 51, -107, -193, 480, -424, 1, -632, 403, 572, -710, -251, 597])
    special_direction = real_theta_ref - real_theta_ref2
    special_direction = special_direction - P @ special_direction


    special_direction_good0 = np.array([2, -3, 6, -1, 1, -2, -4, 10,
                                        -8, 0, -12, 8, 11, -14, -5, 12])
    spec_direction_good1 = np.array([-0, -14, 5, -10, 10, 11, 8, 11,
                                     -5, -6, -14, 5, 11, -13, -11, 11])
    special_direction_good2 = np.array([2, -3, 7, -1, 1, -2, -4, 9,
                                        -8, 0, -12, 8, 11, -13, -5, 12])

    special_direction_good3 = np.array([0.50075807, -0.78763055,  1.62247432, -0.30184803,  0.25290531, -0.5234486,
                                        -0.93948199, 2.33525797, -2.06562868,  0.00486466, -3.07632689,  1.9641191,
                                        2.78731545, -3.4564741,  -1.22235876,  2.90550272])

    special_direction = special_direction / np.sqrt(special_direction.dot(special_direction))
    #special_direction = cross_corr_grad / np.sqrt(cross_corr_grad.dot(cross_corr_grad))
    print(special_direction*10)
    for n in range(-10, 10):
        # new_theta = old_theta + n*np.array([1, 1, 1, 1, 1, 1, 1, 1]) @ P
        new_theta = real_theta_ref + n * 0.1 * special_direction
        # print(new_theta)

        new_gdft = gdft_matrix(16, new_theta)
        gdft_diff = np.linalg.norm(new_gdft - old_gdft)
        new_corrs = thetas_analyzer.get_correlations(new_gdft)
        print(n, new_corrs.avg_auto_corr, gdft_diff)





    '''for theta in sorted_thetas.thetas[0][1:]:
        gdft = gdft_matrix(8, theta)
        avg_auto_corr_gradient = [auto_corr_derivative(sigma, gdft) for sigma in range(8)]
        #print(avg_auto_corr_gradient)
        diff = theta_ref - theta
        maximum = np.min(np.abs(diff))
        print(diff/maximum)'''

    '''sol_spaces = thetas_analyzer.solution_spaces(sorted_thetas, cutoff_ratio=0.8)
    P0 = sol_spaces[0]['projection']
    P1 = sol_spaces[1]['projection']
    print(np.max(np.abs(P1 @ P1 - P)))
    print(P1 @ np.array([1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16]))
    print(P0 @ np.array([1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16]))'''
    #print(np.linalg.matrix_rank(P0, 10**-9))
    #print(np.linalg.matrix_rank(P1, 10 ** -9))
    #print(sol_spaces[0])
    #thetas_group = sorted_thetas.thetas[1]

    '''for label_no in sorted(list(sorted_thetas.thetas.keys())):
        all_thetas = sorted_thetas.thetas[label_no]

        if len(all_thetas) != 0:
            theta_ref = all_thetas[0]
            print("Ratio of what part theta_differences do not depend on the 2-dim subspace generated by dir2 and dir")
            print("for group {}".format(label_no))
            for theta in all_thetas[1:]:
                tot_theta_diff = theta - theta_ref
                #print(tot_theta_diff)
                theta_diff = tot_theta_diff - P @ tot_theta_diff
                #print(theta_diff)
                theta_diff /= np.sqrt(tot_theta_diff.dot(tot_theta_diff))
                ratio = np.sqrt(theta_diff.dot(theta_diff)) / np.sqrt(tot_theta_diff.dot(tot_theta_diff))
                print(ratio)'''

    '''theta_ref = sorted_thetas.thetas[0][0]
    for theta in sorted_thetas.thetas[0][1:]:
        tot_theta_diff = theta - theta_ref
        #print(tot_theta_diff)
        theta_diff = tot_theta_diff - P1 @ tot_theta_diff
        #print(theta_diff)
        theta_diff /= np.sqrt(tot_theta_diff.dot(tot_theta_diff))
        ratio = np.sqrt(theta_diff.dot(theta_diff)) / np.sqrt(tot_theta_diff.dot(tot_theta_diff))
        print(ratio)'''

