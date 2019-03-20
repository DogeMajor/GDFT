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

    theta_collections = extract_thetas_records("../data/", "R_ac_100thetas_8x8__3-7_14_39.json")
    #theta_collections = extract_thetas_records("../data/", "d_ac_100thetas_8x8__3-7_15_10.json")
    #theta_collections = extract_thetas_records("../data/", "R_ac_30thetas_8x8__3-10_13_3.json")

    #theta_collections = extract_thetas_records("../data/", "results_2018-12-24 23_33.json")


    sorted_thetas = thetas_analyzer.sort_thetas(theta_collections.thetas, 6)

    #print(sorted_thetas)

    #thetas0 = sorted_thetas.thetas[0]
    #data_matrix = thetas_analyzer.to_data_matrix(sorted_thetas.thetas[0])
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


    theta8 = theta_collections.thetas[1]
    corr_analyzer = CorrelationAnalyzer(8)


    #The only 'symmetric' theta we found (due to lilitations theta_k in (0, np.pi) ^N
    special_theta = np.array([1.44608874, - 0.24, 3.14159265, 2.35929684,
                              2.35929684, 3.14159265, - 0.24, 1.44608874])

    theta_ref2 = np.array([0.23263316, 1.06778065, 3.05624654, 2.96119473,
                           2.08375977, 0.4239405, 2.96378942, 0.37377238])

    direction = np.array([1, 1, 1, 1,
                          1, 1, 1, 1])

    direction2 = np.array([-7, -5, -3, -1,
                           1, 3, 5, 7])
    #The four directions generate a linear space of dimension 2!!!!
    direction3 = np.array([-2, -1, 0, 1,
                           2, 3, 4, 5])

    direction4 = np.array([4, 3, 2, 1,
                           0, -1, -2, -3])

    #direction5 = np.array([-1, 0, 1, 2, 3, 4, 5, 6])
    direction5 = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    solution_space = np.zeros((8, 4))
    solution_space[:, 0] = direction / np.sqrt(direction.dot(direction))
    solution_space[:, 1] = direction2 / np.sqrt(direction2.dot(direction2))
    solution_space[:, 2] = direction3 / np.sqrt(direction3.dot(direction3))
    solution_space[:, 3] = direction4 / np.sqrt(direction4.dot(direction4))

    u0 = direction / np.sqrt(direction.dot(direction))
    u1 = direction5 - direction5.dot(u0) * u0
    u1 = u1 / np.sqrt(u1.dot(u1))
    P = np.outer(u0, u0) + np.outer(u1, u1)
    print(np.max(np.abs(P @ P - P)))
    print(P @ np.array([1, -2, 3, -4, 5, -6, 7, -8]))
    #should be [1., 0.57142857, 0.14285714, -0.28571429, -0.71428571, -1.14285714, -1.57142857, -2.]
    print(np.linalg.matrix_rank(np.eye(8) - P))

    data_matrix = thetas_analyzer.to_data_matrix(theta_collections.thetas[0:10], subtract_avgs=True)
    U0, sing0, W0 = thetas_analyzer._pca_reduction_svd(data_matrix, cutoff_ratio=0)
    dir0 = W0[:, 0] #R_ac almost constant
    #dir0_new = np.array([-1.9762835, 4.19707068, 2.95672719, -3.88122006,
    #                     1.00000003, 2.42797442, -1.42338032, 2.51663892])

    dir1 = W0[:, 7]  # R_ac almost constant

    #dir1 = dir1 - P @ dir1
    #print(dir1 / -0.1284763)
    #dir1 = np.array([2, -2, -2, 2, 0, 0, 1, -1])

    #dir0_new = np.array([-10, -20, 15, 30,
    #                     0, 0, 0, 50])/ 50
    dir_new = dir1 / np.sqrt(dir1.dot(dir1))
    #dir0_new = dir1
    #print(symm_checker(new_theta, 1))

    #theta_ref = sorted_thetas.thetas[0][0]
    #print(theta_ref)
    '''for theta in sorted_thetas.thetas[0][1:]:
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

    #print(sorted_thetas.thetas)
    thetas_group = [np.array([2.93467664, 0.3384844 , 2.87214115, 1.20613475, 0.32252419,
       0.22130658, 2.20358886, 3.03256426]), np.array([2.9696509 , 0.32219672, 2.80460713, 1.08736893, 0.15248202,
       0.        , 1.93102203, 2.70875109]), np.array([3.14150297, 0.50597848, 3.00032683, 1.29500184, 0.37205216,
       0.23150659, 2.17446601, 2.96410783]), np.array([2.85346825, 0.24653422, 2.76946155, 1.09272548, 0.19836428,
       0.08639467, 2.05793914, 2.87617182]), np.array([2.64995925, 0.1094916 , 2.69888498, 1.08861413, 0.26072341,
       0.21521909, 2.25322855, 3.13791887]), np.array([3.11428113, 0.49839706, 3.01236833, 1.32667583, 0.42337081,
       0.30244392, 2.2650365 , 3.07431384]), np.array([2.97236851, 0.3847211 , 2.92693977, 1.26949081, 0.39442552,
       0.30174438, 2.29258057, 3.1300973 ]), np.array([2.90780283, 0.31334572, 2.84876456, 1.18452025, 0.30265909,
       0.20317343, 2.18723403, 3.01791636]), np.array([2.73064916, 0.14368992, 2.68659138, 1.02982502, 0.15544191,
       0.06345344, 2.05497724, 2.89317578]), np.array([2.98336504, 0.33398594, 2.81447009, 1.09527923, 0.15848194,
       0.00406954, 1.93317593, 2.70895432]), np.array([3.03389687, 0.38015566, 2.85628297, 1.13274342, 0.19158676,
       0.03281154, 1.95755342, 2.72897963])]

    thetas_group2 = [np.array([0.23263316, 1.06778065, 3.05624654, 2.96119473, 2.08375977, 0.4239405, 2.96378942, 0.37377238]),
                     np.array([0.12853125, 0.96144711, 2.94767889, 2.85038891, 1.97072153, 0.30866741, 2.8462803, 0.25403236]),
                     np.array([0.43171271, 1.20998793, 3.14159265, 2.9896899, 2.05536774, 0.33869695, 2.82167874, 0.17477883]),
                     np.array([0.4590644, 1.22344731, 3.14116473, 2.97536152, 2.02716572, 0.29658784, 2.76567391, 0.10491548]),
                     np.array([0.18417949, 0.97587018, 2.92086764, 2.7823677, 1.86144584, 0.15818738, 2.65457865, 0.02110813]),
                     np.array([0.09761061, 0.91594684, 2.88757434, 2.77571619, 1.88147829, 0.2048429, 2.727869, 0.12101726]),
                     np.array([1.51708874, 0., 3.14159265, 2.43028616, 2.43029684, 3.14159265, 0., 1.51711487]),
                     np.array([0.84404373, 2.5896863, 3.14159265, 0., 3.14159265, 1.27085174, 1.35837218, 0.22956873])]

    from analyzer import SortedThetas
    sorted_thetas = SortedThetas(thetas={0: thetas_group, 1: thetas_group2}, labels=[], histogram={})
    print(sorted_thetas.thetas[0])

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

    theta_ref = thetas_group[0]
    for theta in thetas_group[1:]:
        tot_theta_diff = theta - theta_ref
        # print(tot_theta_diff)
        theta_diff = tot_theta_diff - P @ tot_theta_diff
        # print(theta_diff)
        theta_diff /= np.sqrt(tot_theta_diff.dot(tot_theta_diff))
        ratio = np.sqrt(theta_diff.dot(theta_diff)) / np.sqrt(tot_theta_diff.dot(tot_theta_diff))
        print(ratio)


    example_theta = np.array([0.43171271, 1.20998793, 3.14159265, 2.9896899,
                              2.05536774, 0.33869695, 2.82167874, 0.17477883])

    theta_ref = theta_collections.thetas[0]
    old_gdft = gdft_matrix(8, special_theta)
    special_direction = np.array([0, 0, -1, 0, 0, -1, 0, 0])

    '''for n in range(-5, 5):

        constr_gdft = gdft_matrix(8, special_theta+n*0.1*direction)
        correlations = corr_analyzer.get_correlations(constr_gdft)
        gdft_diff = np.linalg.norm(old_gdft - constr_gdft)
        print(special_theta + n*0.1*direction)
        print(n, correlations.avg_auto_corr, gdft_diff)'''


    #polar_plot_angles(new_theta)

    #print(theta_init - example_theta)
    #print(theta_init + 10*0.015*direction)

    #for theta in sorted_thetas.thetas[0][0:10]:
    #    print(theta)
    #polar_plot_numbered_angles(special_theta)
    #plt.show()

    #Correlations(max_auto_corr=0.3814517374888245, avg_auto_corr=(1.0209859493694915+0j),
    #max_cross_corr=0.5286804616397811, avg_cross_corr=(0.854144864375787+0j), avg_merit_factor=(0.9794454082522375+0j))'''