from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from utils import extract_thetas_records, seq_norm, approximate_matrix, approximate_phases
from gdft import dft_matrix, gdft_matrix, two_param_gdft_matrix
from analyzer import ThetasAnalyzer
from correlations import CorrelationAnalyzer, Correlation

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


kmean_thetas = [np.array([2.9135797, 0.39698846, 2.63539188, 1.42586124,
                          0.32580239, 0.41098031, 2.19474127, 3.05086212]),
                np.array([2.92536849, 2.11414487, 0.14960736, 0.26858388,
                          1.16994527, 2.85369467, 0.33776914, 2.95171189]),
                np.array([0.27610676, 1.03383679, 2.89123087, 2.93963802,
                          1.90278413, 0.76288471, 2.78617887, 0.4474727]),
                np.array([1.79351973, 2.50482738, 1.67077691, 0.23710056,
                          2.33149689, 0.42360577, 1.68394482, 1.38386787]),
                np.array([0.25785108, 2.86088575, 0.33405658, 2.00689391,
                          2.89734625, 3.00541467, 1.02996681, 0.20784047]),
                np.array([4.31449187, 1.12524368, 1.80579287, -0.5236294,
                          0.56513176, 1.39744013, 0.64624049, 4.16964116])]

new_kmean_thetas = [[0.29941173, 2.89847069, 0.36766799, 2.03652784,
                     2.92300659, 3.02710112, 1.04767779, 0.22157399],
                    [0.24725148, 1.06828249, 3.04262742, 2.9334619,
                     2.04191348, 0.36797856, 2.89371675, 0.28957994],
                    [3.01502533, 2.19267569, 0.23495331, 0.35717945,
                     1.25005117, 2.92609014, 0.39601341, 3.01691177],
                    [1.42282673, 0.95521555, 2.6179977, 1.73874107,
                     2.26234306, 2.30620117, 0.74999046, 1.32040115],
                    [3.25837994, 1.03165323, 1.28296113, 0.72212853,
                     0.09730907, 2.731586, 0.59759348, 3.24538009],
                    [3.01110803, 0.40223826, 2.92323331, 1.24455925,
                     0.3482716, 0.23436835, 2.20397707, 3.02027523]]


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
    
    #sorted_thetas = thetas_analyzer.sort_thetas(theta_collections.thetas, 6)
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


    def derivative(gdft, sigma, alpha, beta, pos_mu):
        N = gdft.shape[0]
        mu = pos_mu - N + 1
        is_positive = 0 < mu <= N - 1
        is_negative = 1 - N <= mu <= 0

        def pos_val(sigma):
            in_interval_1 = sigma <= min(N - mu - 1, mu)
            in_interval_2 = mu < sigma < N - mu - 1
            in_interval_3 = N - mu - 1 < sigma < mu
            in_interval_4 = sigma >= max(N - mu - 1, mu)

            if in_interval_1:
                return (1j / N) * gdft[alpha, sigma] * np.conjugate(gdft[beta, sigma + mu])
            if in_interval_2:
                return (1j / N) * (gdft[alpha, sigma] * np.conjugate(gdft[beta, sigma + mu])
                                   - gdft[alpha, sigma - mu] * np.conjugate(gdft[beta, sigma]))
            if in_interval_3:
                return 0
            if in_interval_4:
                return (-1j / N) * gdft[alpha, sigma - mu] * np.conjugate(gdft[beta, sigma])

        def neg_val(sigma):
            in_interval_1 = sigma <= min(N + mu - 1, -mu - 1)
            in_interval_2 = -mu <= sigma < N + mu - 1
            in_interval_3 = N + mu - 1 < sigma < -mu
            in_interval_4 = sigma >= max(N + mu - 1, -mu)
            # print(in_interval_1, in_interval_2, in_interval_3, in_interval_4)
            if in_interval_1:
                return (-1j / N) * gdft[alpha, sigma - mu] * np.conjugate(gdft[beta, sigma])
            if in_interval_2:
                return (1j / N) * (gdft[alpha, sigma] * np.conjugate(gdft[beta, sigma + mu])
                                   - gdft[alpha, sigma - mu] * np.conjugate(gdft[beta, sigma]))
            if in_interval_3:
                return 0
            if in_interval_4:
                return (-1j / N) * gdft[alpha, sigma] * np.conjugate(gdft[beta, sigma + mu])

        if is_positive:
            return pos_val(sigma)

        if is_negative:
            return neg_val(sigma)


    def gradient(gdft, alpha, beta, mu):
        N = gdft.shape[0]
        derivatives = [derivative(gdft, sigma, alpha, beta, mu) for sigma in range(N)]
        return np.array(derivatives)

    def derivative_Rac(sigma, gdft):
        N = gdft.shape[0]
        result = 0
        corr_tensor = Correlation(gdft).correlation_tensor()
        conj_tensor = np.conjugate(corr_tensor)
        for alpha in range(N):
            for mu in range(1, N):
                first_term = derivative(gdft, sigma, alpha, alpha, mu)*conj_tensor[alpha, alpha, mu]
                second_term = np.conjugate(derivative(gdft, sigma, alpha, alpha, mu)) * corr_tensor[alpha, alpha, mu]
                result += first_term + second_term
                #print(alpha, mu, result)
        return result/N

    def grad_Rac(gdft):
        N = gdft.shape[0]
        derivatives = [derivative_Rac(sigma, gdft) for sigma in range(N)]
        return np.array(derivatives)

    def difference(analyzer, theta, sigma, corr_name, h=0.00001):
        N = theta.shape[0]
        #print(sigma, h)
        old_gdft = gdft_matrix(N, theta)
        old_corr_tensor = Correlation(old_gdft).correlation_tensor()
        old_corr = analyzer._corr_fns[corr_name](old_corr_tensor)
        new_theta = deepcopy(theta)
        new_theta[sigma] += h
        new_gdft = gdft_matrix(N, new_theta)
        new_corr_tensor = Correlation(new_gdft).correlation_tensor()
        new_corr = analyzer._corr_fns[corr_name](new_corr_tensor)
        return (new_corr-old_corr)/h

    def grad(analyzer, theta, corr_name, step=0.00001):
        diffs = [difference(analyzer, theta, index, corr_name, h=step) for index, _ in enumerate(theta)]
        return np.array(diffs)


    def diff_c_tensor(theta, sigma, h=0.00001):
        N = theta.shape[0]
        #print(sigma, h)
        old_gdft = gdft_matrix(N, theta)
        old_corr_tensor = Correlation(old_gdft).correlation_tensor()

        new_theta = deepcopy(theta)
        new_theta[sigma] += h
        new_gdft = gdft_matrix(N, new_theta)
        new_corr_tensor = Correlation(new_gdft).correlation_tensor()
        return (new_corr_tensor-old_corr_tensor)/h

    def diff_avg_corr(theta, sigma, h=0.00001):
        N = theta.shape[0]
        ct_diff = diff_c_tensor(theta, sigma, h=h)
        conj_ct_diff = np.conjugate(ct_diff)
        gdft = gdft_matrix(N, theta)
        ct = Correlation(gdft).correlation_tensor()

        def compute_term(a, mu):
            return ct_diff[a, a, mu]*np.conjugate(ct[a, a, mu])+ct[a, a, mu]*conj_ct_diff[a, a, mu]

        result = 0
        for alpha in range(N):
            for mu in range(N-1):
                result += compute_term(alpha, mu)
            for mu in range(N, 2*N-1):
                result += compute_term(alpha, mu)

        return result/N

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
    diffs = np.array([difference(corr_analyzer, theta8, ind, "avg_auto_corr", h=0.00001) for ind in range(8)])
    print(diffs)
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
    avg_corr_diffs = np.array([diff_avg_corr(theta8, ind, h=0.00001) for ind in range(8)])
    print(avg_corr_diffs)
    #theta2 = theta_collections.thetas[10]
    #print("avg_auto_corr", grad(corr_analyzer, theta2, "max_auto_corr", step=0.0001))
    #diffs = [difference(corr_analyzer, theta8, index, "avg_auto_corr", h=0.001) for index in range(8)]
    #print(diffs)
    #print("max_cross_corr", grad(corr_analyzer, theta8, "max_cross_corr", h=0.00001))
    #print("avg_cross_corr", grad(corr_analyzer, theta8, "avg_cross_corr", h=0.00001))
    #print(symm_checker(new_theta, 1))

    '''gdft = gdft_matrix(8, theta3)
    avg_auto_corr_gradient = grad_Rac(gdft)
    print(avg_auto_corr_gradient)
    rac_der1 = derivative_Rac(1, gdft)
    print(rac_der1)'''


    poly_coeff_8 = np.array([-7.47998864e-03, 1.73916258e-01, -1.61020449e+00, 7.60456544e+00,
                             -1.93127379e+01, 2.45158151e+01, -1.05428434e+01, 2.47251476e-01])

    poly_coeff_4 = np.array([1.04719702, -4.05332898, 2.84656905, 2.63384441])
    '''
    #THETAS8 = np.array([-3.12, -3.38, -3.2, -1.86, -1.27, 0.06, 0.25, -0.01])
    THETAS8 = np.array([1.637, -0.79, -0.54, 2.01, 1.59, -0.83, 1.73, 2.44])

    #gdft = gdft_matrix(8, THETAS8)
    gdft = two_param_gdft_matrix(8, np.zeros(8), THETAS8)
    correlations = thetas_analyzer.get_correlations(dft_matrix(8))
    print(correlations)
    '''
    #Correlations(max_auto_corr=0.3814517374888245, avg_auto_corr=(1.0209859493694915+0j),
    #max_cross_corr=0.5286804616397811, avg_cross_corr=(0.854144864375787+0j), avg_merit_factor=(0.9794454082522375+0j))'''