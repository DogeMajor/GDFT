import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from utils import extract_thetas_records, seq_norm, approximate_matrix, approximate_phases, Thetas
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


def center_thetas(all_thetas, index=0, epsilon=1e-10):#Epsilon is needed to keep the data matrix pos. definite!!!
    return [(theta - theta[index] + epsilon) % (2*np.pi) for theta in all_thetas]


def closest_matches(all_thetas):
    amount = len(all_thetas)
    shortest_distance = np.linalg.norm(all_thetas[0] - all_thetas[1])
    dist = 0
    index_pair = (0, 1)
    for index in range(amount):
        for n in range(index+1, amount):
            dist = np.linalg.norm(all_thetas[index] - all_thetas[n])
            if dist <= shortest_distance:
                shortest_distance = dist
                index_pair = (index, n)

    return index_pair, shortest_distance


def theta_diffs(all_thetas):
    amount = len(all_thetas)
    diffs = np.zeros((amount, amount, len(all_thetas[0])))
    print("diffs shape ", diffs.shape)
    for index in range(amount):
        for n in range(index+1, amount):
            #diffs.append(all_thetas[index] - all_thetas[n])
            diffs[index, n, :] = all_thetas[index] - all_thetas[n]

    return diffs


def theta_directions(theta_diffs):
    amount = theta_diffs.shape[0]
    differences = np.zeros(theta_diffs.shape)
    print(differences.shape)
    for index in range(amount):
        for n in range(index+1, amount):
            diff = theta_diffs[index, n, :]
            differences[index, n, :] = diff / np.sqrt(diff.dot(diff))

    return differences


def subtract_projections(theta_dirs, projection):
    amount = theta_dirs.shape[0]
    dirs = np.zeros(theta_dirs.shape)
    for index in range(amount):
        for n in range(index+1, amount):
            diff = theta_dirs[index, n, :] - projection @ theta_dirs[index, n, :]
            dirs[index, n, :] = diff / np.sqrt(diff.dot(diff))

    return dirs


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
    #theta_collections = extract_thetas_records("../data/", "R_ac_30thetas_16x16__3-4_21_58.json")
    theta_collections = extract_thetas_records("../data/", "R_ac_100thetas_16x16__3-21_14_11.json")
    centered_thetas = center_thetas(theta_collections.thetas)
    #theta_collections = extract_thetas_records("../data/", "results_2018-12-24 23_33.json")

    sorted_thetas = thetas_analyzer.sort_thetas(centered_thetas, 6)
    #print(sorted_thetas)

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

    thetas16 = np.array([4.12065179, 3.6280368, 3.63853898, 1.98545421, 1.19573082,
                         2.431672, 3.57920976, 3.4751789, 6.14732732, 4.1644376,
                         2.32034156, 5.32808867, 0.11059888, 2.6664847, 5.57201023, 1.74524767])

    second_thetas16 = np.array([2.37625422e+00, 6.21340493e-01, 8.52913985e-04, 3.77984053e+00,
                                3.90729155e+00, 1.54190310e+00, 3.15592357e+00, 3.18682849e+00,
                                8.32899617e-01, 3.28850540e+00, 1.36839105e+00, 4.30070095e+00,
                                3.49656598e-02, 7.60257752e-01, 7.59276663e-01, 2.11501830e+00])

    third_thetas16 = np.array([8.56020129e-04, 1.65289296e+00, 2.80292025e+00, 3.64906793e+00,
                               5.02269818e+00, 1.58942683e+00, 4.88402038e+00, 2.68697355e-01,
                               3.64885887e+00, 1.35446580e+00, 6.21213519e+00, 3.25536125e+00,
                               1.66153796e+00, 1.78281113e+00, 1.23107396e+00, 5.98980120e-01])

    fourth_thetas16 = np.array([0.51421526, 4.90164146, 3.50782438, 2.41788531,
                                0.80041219, 3.98995115, 0.45157117, 4.82309165,
                                1.19909394, 3.24968651, 4.43143693, 0.86125633,
                                2.21129148, 1.84630953, 2.15420136, 2.5424888 ])

    corrs = corr_analyzer.get_correlations(gdft_matrix(16, thetas16))

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

    thetas_group0 = [np.array([1.00000000e-10, 5.14234478e+00, 2.64467663e+00, 8.73458611e-01,
                               3.94364859e-01, 8.30033114e-01, 2.69658305e+00, 2.65548237e+00,
                               4.08816075e+00, 1.62242262e+00, 7.39899058e-01, 2.16103190e+00,
                               6.07508972e+00, 1.07433989e+00, 4.24048985e+00, 6.27221384e+00]),
                     np.array([1.00000000e-10, 2.88524546e+00, 3.26574949e+00, 5.66222922e+00,
                               5.59120672e-01, 2.61987547e+00, 9.44225481e-02, 3.76243370e+00,
                               4.66100962e+00, 2.79029232e+00, 8.29780887e-01, 5.66209787e-01,
                               1.18339920e+00, 5.84155288e-01, 2.00083922e+00, 9.12867702e-01]),
                     np.array([1.00000000e-10, 2.19165483e+00, 3.24893320e+00, 6.18983266e+00,
                               2.15990261e-01, 3.01815788e+00, 1.84087252e+00, 2.24680983e+00,
                               5.03744288e+00, 3.01871641e+00, 5.37553132e+00, 2.88004391e+00,
                               1.29931798e+00, 1.01074043e+00, 1.44856814e+00, 5.29554597e-01]),
                     np.array([1.00000000e-10, 1.67444915e+00, 3.42920762e+00, 5.85694350e+00,
                               2.86402997e-01, 5.91934417e+00, 5.17174876e-01, 5.29230875e-01,
                               6.21578348e+00, 3.90698466e+00, 3.22495598e+00, 2.09811897e+00,
                               5.77809227e+00, 2.64759473e+00, 6.10416582e+00, 3.77946742e+00]),
                     np.array([1.00000000e-10, 2.15043562e+00, 3.16638306e+00, 6.06610184e+00,
                               5.09869193e-02, 2.81195811e+00, 1.59343586e+00, 1.95809883e+00,
                               4.70751556e+00, 2.64755638e+00, 4.96312769e+00, 2.42644198e+00,
                               8.04432959e-01, 4.74668932e-01, 8.71202283e-01, 6.19415538e+00])]

    thetas_group1 = [np.array([1.00000000e-10, 1.80713691e+00, 4.57244815e+00, 4.58930796e+00,
                                1.46562269e+00, 3.01758088e+00, 5.33893138e-01, 3.99210228e+00,
                                7.93942130e-01, 1.27773167e-01, 4.69243856e+00, 3.25334623e+00,
                                4.58989594e+00, 4.66886430e+00, 5.20930055e+00, 5.47566791e+00]),
                     np.array([1.00000000e-10, 3.53376040e+00, 3.35560450e-01, 3.76999104e+00,
                            2.41146511e+00, 5.39411057e+00, 6.62106396e-01, 2.35221674e+00,
                            5.67043450e+00, 5.48159393e+00, 4.04127056e+00, 2.51254152e+00,
                            3.00944560e+00, 4.36971898e+00, 4.06641958e+00, 4.26614425e+00]),
                     np.array([1.00000000e-10, 2.59548631e+00, 1.46178083e+00, 4.47250874e+00,
                            3.06398266e+00, 8.23617737e-02, 3.86966039e-01, 5.65206079e-01,
                            2.10681099e+00, 7.50052594e-01, 6.08430870e-01, 4.16197130e-02,
                            3.24142262e+00, 4.56942399e+00, 1.70525898e+00, 3.01956603e+00]),
                     np.array([1.00000000e-10, 3.86842615e+00, 1.95672169e+00, 4.66465701e+00,
                            1.95275513e+00, 1.26650077e+00, 4.91801456e-01, 4.74884993e+00,
                            5.49877579e+00, 1.59369291e+00, 4.11067757e+00, 5.19620679e+00,
                            4.33973436e+00, 4.97347399e+00, 5.95675543e+00, 2.07927622e-01]),
                     np.array([1.00000000e-10, 4.74434631e+00, 3.12523670e+00, 8.33127600e-01,
                            2.56115417e-01, 1.04192925e+00, 2.96596092e-01, 4.20171155e-01,
                            1.15229804e+00, 3.59674080e+00, 4.41450428e+00, 5.67686046e+00,
                            2.13236481e+00, 5.39846012e+00, 2.07760913e+00, 4.53790362e+00]),
                     np.array([1.00000000e-10, 2.61543579e-01, 2.11881372e-02, 5.75997786e+00,
                            5.74298540e+00, 9.19179753e-01, 2.82327410e+00, 3.10073416e+00,
                            5.09032377e+00, 1.40545071e+00, 4.87269126e+00, 5.25433394e-01,
                            3.82425101e+00, 2.55499913e+00, 6.12829303e-01, 4.87347187e+00]),
                     np.array([1.00000000e-10, 1.40056489e+00, 3.30319495e+00, 5.50970114e+00,
                            9.05564851e-01, 1.10830904e+00, 8.66348265e-01, 2.25116273e+00,
                            1.92365762e+00, 9.87552412e-01, 5.46568394e+00, 5.19198666e+00,
                            3.55527657e+00, 2.03421603e-01, 3.80783534e+00, 1.20937503e+00]),
                     np.array([1.00000000e-10, 3.82287468e+00, 1.86561001e+00, 4.52806872e+00,
                            1.77059787e+00, 1.03890362e+00, 2.18668143e-01, 4.43019021e+00,
                            5.13464423e+00, 1.18405622e+00, 3.65545460e+00, 4.69544581e+00,
                            3.79350617e+00, 4.38165082e+00, 5.31952180e+00, 5.80826577e+00]),
                     np.array([1.00000000e-10, 4.01273274e+00, 6.07819756e-01, 5.36981701e+00,
                            1.21695273e+00, 5.84020598e+00, 2.00839086e-01, 2.42775186e+00,
                            7.56305386e-01, 5.58630862e-01, 4.73649384e+00, 4.06206804e+00,
                            4.30240941e+00, 5.83489537e+00, 1.81057723e+00, 2.71261404e+00]),
                     np.array([1.00000000e-10, 6.80766058e-01, 8.58553823e-01, 2.69992361e+00,
                            3.67783553e+00, 2.63023111e+00, 1.67093534e+00, 1.96321041e+00,
                            5.76247141e+00, 1.65046902e+00, 3.68268200e+00, 8.63308606e-01,
                            6.26897821e+00, 3.90129240e+00, 1.18415097e+00, 5.19899308e+00]),
                     np.array([1.00000000e-10, 2.65445638e+00, 1.60163433e-01, 2.06537932e+00,
                            3.19011017e-01, 2.36292722e+00, 2.10309490e+00, 2.60161756e-01,
                            2.31562100e+00, 2.89727064e+00, 5.38658714e+00, 1.61831249e-01,
                            3.05483318e-01, 5.44014781e+00, 3.56532224e+00, 3.04722055e+00]),
                     np.array([1.00000000e-10, 5.53753035e+00, 4.87221510e+00, 4.87994474e+00,
                            3.17255152e+00, 1.02193283e-01, 4.84632418e+00, 2.43839339e+00,
                            5.70500997e+00, 9.76092368e-01, 4.15718317e+00, 6.10337871e-01,
                            1.87042457e+00, 2.60298209e+00, 3.63949650e+00, 5.17800895e+00]),
                     np.array([1.00000000e-10, 4.32467709e+00, 8.13659705e-01, 4.95337513e+00,
                            1.51412770e+00, 2.06989221e+00, 1.28435972e+00, 2.43444147e+00,
                            2.30334695e+00, 7.74378773e-01, 5.48604816e+00, 5.78808720e+00,
                            1.85760945e+00, 2.91933552e+00, 5.84488263e+00, 1.79039272e-01]),
                     np.array([1.00000000e-10, 2.06092101e+00, 2.98737916e+00, 5.79756277e+00,
                            5.97610286e+00, 2.36436937e+00, 1.05636034e+00, 1.33149161e+00,
                            3.99141116e+00, 1.84193622e+00, 4.06798952e+00, 1.44176557e+00,
                            6.01348072e+00, 5.59413271e+00, 5.90120323e+00, 4.85145266e+00]),
                     np.array([1.00000000e-10, 1.30673212e+00, 3.11562421e+00, 5.22823041e+00,
                            5.30302819e-01, 6.39265630e-01, 3.03385824e-01, 1.59447060e+00,
                            1.17316599e+00, 1.43222376e-01, 4.52760529e+00, 4.16000460e+00,
                            2.42947064e+00, 5.26692795e+00, 2.49440056e+00, 6.08533512e+00]),
                     np.array([1.00000000e-10, 1.44269988e+00, 2.43654155e+00, 3.77987651e+00,
                            3.30204908e-01, 1.22174407e+00, 6.81861099e-01, 2.80811735e-01,
                            1.50793391e+00, 5.51117566e+00, 1.97972595e+00, 4.64308326e+00,
                            3.04884519e+00, 2.31801117e+00, 6.20676349e+00, 4.31538828e+00]),
                     np.array([1.00000000e-10, 2.19165483e+00, 3.24893320e+00, 6.18983266e+00,
                            2.15990261e-01, 3.01815788e+00, 1.84087252e+00, 2.24680983e+00,
                            5.03744288e+00, 3.01871641e+00, 5.37553132e+00, 2.88004391e+00,
                            1.29931798e+00, 1.01074043e+00, 1.44856814e+00, 5.29554597e-01]),
                     np.array([1.00000000e-10, 5.45299436e+00, 5.07163294e+00, 4.34079672e+00,
                            2.11979446e+00, 1.84072929e+00, 2.99314686e+00, 4.00664684e+00,
                            3.39205066e+00, 1.26661250e-03, 4.14522393e+00, 2.09435097e+00,
                            4.30111267e+00, 5.64443154e+00, 2.36814666e+00, 4.87203424e+00]),
                     np.array([1.00000000e-10, 1.67444915e+00, 3.42920762e+00, 5.85694350e+00,
                            2.86402997e-01, 5.91934417e+00, 5.17174876e-01, 5.29230875e-01,
                            6.21578348e+00, 3.90698466e+00, 3.22495598e+00, 2.09811897e+00,
                            5.77809227e+00, 2.64759473e+00, 6.10416582e+00, 3.77946742e+00]),
                     np.array([1.00000000e-10, 1.71780890e+00, 2.98678804e+00, 4.60524237e+00,
                            1.43069145e+00, 2.59725590e+00, 2.33252899e+00, 2.20662335e+00,
                            3.70888249e+00, 1.70407897e+00, 4.73094077e+00, 1.38622955e+00,
                            6.70803257e-02, 5.89454630e+00, 3.77527380e+00, 2.15898080e+00]),
                     np.array([1.00000000e-10, 2.15043562e+00, 3.16638306e+00, 6.06610184e+00,
                            5.09869193e-02, 2.81195811e+00, 1.59343586e+00, 1.95809883e+00,
                            4.70751556e+00, 2.64755638e+00, 4.96312769e+00, 2.42644198e+00,
                            8.04432959e-01, 4.74668932e-01, 8.71202283e-01, 6.19415538e+00])]


    theta_avg0 = sum(thetas_group0) / len(thetas_group0)

    theta_ref = thetas_group1[14]
    data_matrix = thetas_analyzer.to_data_matrix(thetas_group1)
    U1, sing_vals1, W1 = thetas_analyzer._pca_reduction_svd(data_matrix, cutoff_ratio=0.15)
    print(W1.shape)
    P1 = np.outer(W1[0, :], W1[0, :]) + np.outer(W1[1, :], W1[1, :]) + 0*np.outer(W1[2, :], W1[2, :]) + 0*np.outer(W1[3, :], W1[3, :])

    print("P1 -  P", np.linalg.norm(P1 - P))
    u11, svals11, w11 = thetas_analyzer._pca_reduction_svd(P1 - P, cutoff_ratio=0.5)
    print(w11)
    cross_corr_grad = auto_corr_gradient(gdft_matrix(16, theta_ref))
    cross_corr_grad /= np.sqrt(cross_corr_grad.dot(cross_corr_grad))
    P_grad = np.outer(cross_corr_grad, cross_corr_grad)

    from sequencefinder import SequenceFinder
    seq_finder = SequenceFinder()

    indices, dist = closest_matches(thetas_group1)
    shortest_diff = thetas_group1[10] - thetas_group1[19]
    theta_diff = shortest_diff - P @ shortest_diff
    print(np.linalg.norm(shortest_diff))
    theta_diffs = theta_diffs(thetas_group1)

    print(theta_diffs[0, 1, 2])
    print(theta_diffs.shape)
    theta_directions = theta_directions(theta_diffs)
    direction_mat = subtract_projections(theta_directions, P)
    print("direction")
    print(direction_mat[1, 13, :])
    amount = direction_mat.shape[0]
    for i in range(amount):
        for j in range(i+1, amount):
            res = direction_mat[1, 2, :].dot(direction_mat[i, j, :])
            if np.abs(res) > 0.7:
                print("cos between 1,2 and {}, {}: {}".format(i, j, res))


    dir_spessu = theta_directions[14, 16, :]

    dir_spessu = dir_spessu - P @ dir_spessu
    #print(dir_spessu)

    #[0, 6, -11, -8, -14, 13, -1, 5, 8, 16, 1, 6, -11, -4 -6, -1]
    #thetas_group = [thetas16, second_thetas16, third_thetas16, fourth_thetas16]
    '''for theta in thetas_group1[0:0] + thetas_group1[1:]:

        tot_theta_diff = theta - theta_ref
        # print(tot_theta_diff)
        theta_diff = tot_theta_diff - P @ tot_theta_diff
        #theta_diff = tot_theta_diff - P_grad @ tot_theta_diff
        print(theta_diff)
        print(seq_finder._to_appr_integers(theta_diff))
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

    old_gdft = gdft_matrix(16, theta_ref)
    special_direction = np.array([9.3, 16.2, -14.8, -13.2, -2.4, 4, 3.3, -11.2,
                                  -7.4, 12.7, 7.7, -8.1, 3, -3.3, -2, 6.2])

    special_direction = np.array([0, 6, -11, -8, -14, 13, -1, 5,
                                  8, 16, 1, 6, -11, -4, -6, -1])

    special_direction = direction_mat[14, 16, :]
    special_direction = np.array([-3.68171308e-16, -1.65027466e-01, -5.50795708e-02,  5.07552294e-02,
                                  4.35740960e-01, -7.56627669e-02,  2.19634396e-01,  4.73383166e-03,
                                  3.41771413e-01, -9.72340613e-02, -1.04846083e-01, -5.91163249e-01,
                                  1.51504222e-01, -1.29988536e-02, -3.30977397e-01,  3.22338302e-01])

    special_direction = np.array([4.53776560e-01,  4.18489269e-01, 3.53241003e-01,  2.88440895e-01,
                                  1.93224848e-01, 1.95678285e-01,  1.10234593e-01,  8.03814127e-02,
                                  -9.61026376e-03, -1.50452506e-02, -6.74843335e-02, -6.77642970e-02,
                                  -2.01952904e-01, -2.37297332e-01, -2.55919263e-01, -3.80372211e-01])
    print("spec direction", special_direction)
    special_direction_good0 = np.array([2, -3, 6, -1, 1, -2, -4, 10,
                                        -8, 0, -12, 8, 11, -14, -5, 12])
    spec_direction_good1 = np.array([-0, -14, 5, -10, 10, 11, 8, 11,
                                     -5, -6, -14, 5, 11, -13, -11, 11])
    special_direction_good2 = np.array([2, -3, 7, -1, 1, -2, -4, 9,
                                        -8, 0, -12, 8, 11, -13, -5, 12])

    special_direction_good3 = np.array([0.50075807, -0.78763055,  1.62247432, -0.30184803,  0.25290531, -0.5234486,
                                        -0.93948199, 2.33525797, -2.06562868,  0.00486466, -3.07632689,  1.9641191,
                                        2.78731545, -3.4564741,  -1.22235876,  2.90550272])

    special_direction_good4 = np.array([-1.46300874 + 0.j, 1.50425889 + 0.j, 3.10830139 + 0.j, -0.80748358 + 0.j,
                                        -0.00753234 + 0.j, -0.34452473 + 0.j, 1.89849061 + 0.j, -3.31834563 + 0.j,
                                        1.81219855 + 0.j, -1.02808251 + 0.j, 4.84030468 + 0.j, -0.23577141 + 0.j,
                                        1.68131092 + 0.j, 0.92955698 + 0.j, 0.63982304 + 0.j, -1.6239422 + 0.j])

    special_direction_good5 = np.array([-0.67552351, -4.14049372,  0.11485832,  4.31673764, -0.77178368 , 4.42841466,
                                        -2.83737915, -2.78129705,  1.4755026,   1.63536733,  1.83878764, -0.70625679,
                                        -0.93741589,  0.93576182,  1.22910838, -3.12438859])
    #special_direction = special_direction - P @ special_direction
    special_direction = special_direction / np.sqrt(special_direction.dot(special_direction))
    print("u0*dir", u0.dot(special_direction))
    print("u1*dir", u1.dot(special_direction))
    #special_direction = cross_corr_grad / np.sqrt(cross_corr_grad.dot(cross_corr_grad))
    #print("gradient times special dir", special_direction.dot(cross_corr_grad))
    for n in range(-10, 10):
        # new_theta = old_theta + n*np.array([1, 1, 1, 1, 1, 1, 1, 1]) @ P
        new_theta = theta_ref + n * 0.1 * special_direction
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

