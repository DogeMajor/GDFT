import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from utils import extract_thetas_records, seq_norm, approximate_matrix, approximate_phases
from gdft import dft_matrix, gdft_matrix
from analyzer import ThetasAnalyzer

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
    #theta_collections = extract_thetas_records("../data/", "10thetas_16x16__12-27_15_38.json")
    #theta_collections = extract_thetas_records("../data/", "30thetas_16x16__1-1_21_14.json")
    #theta_collections = extract_thetas_records("../data/", "10thetas_16x16__12-27_11_58.json")
    #theta_collections = thetas = extract_thetas_records("../data/", "10thetas_16x16__12-26_19_4.json")
    #theta_collections = thetas = extract_thetas_records("../data/", "100thetas_4x4__12-26_16_6.json")
    theta_collections = extract_thetas_records("../data/", "100thetas12-26_1_26.json")
    #theta_collections = extract_thetas_records("../data/", "results_2018-12-24 23_33.json")
    thetas_analyzer = ThetasAnalyzer(8)
    sorted_thetas = thetas_analyzer.sort_thetas(theta_collections.thetas, 6)
    thetas0 = sorted_thetas.thetas[0]
    results0 = thetas_analyzer.fit_polynomes(sorted_thetas.thetas[0], 7)
    results = thetas_analyzer.fit_polynomes(theta_collections.thetas, 7)
    #for poly, theta in zip(results0.polynomes, results0.theta_vecs):
    #    plot_fitted_polynome(poly, theta)

    cov_mat = thetas_analyzer.get_covariance_matrix(0, sorted_thetas)
    #print(cov_mat)
    cov_mat, avgs = thetas_analyzer.get_gaussian_params(0, sorted_thetas)
    print(cov_mat, avgs)
    print(linalg.det(cov_mat))
    #print(linalg.inv(cov_mat))
    eigen_values, eigen_vectors = linalg.eig(cov_mat)
    print(cov_mat.shape)
    print(np.linalg.matrix_rank(cov_mat))
    print("eigenvalues", eigen_values)
    print("eigenvectors", eigen_vectors)

    print(thetas_analyzer.entropy(cov_mat))
    new_eigen_values, new_eigen_vectors = thetas_analyzer._pca_reduction(cov_mat, cutoff_ratio=0.01)
    print(new_eigen_values, new_eigen_vectors)
    print(eigen_vectors.dot(np.diagflat(eigen_values).dot(eigen_vectors.T)))
    #print(eigen_vectors * eigen_values)
    print(cov_mat)
    #print(la)
    #plot_fitted_polynome(poly, results0.theta_vecs[0])
    #print(sorted_thetas)
    #print(fitted_polynomes)

    #best_thetas = find_best_orderings(theta_collections)
    #print(best_thetas)

    #for item in best_thetas[0:]:
    #    print(np.argsort(item[0]))
    #    print(orderings_dist(item[0]))

    '''for thetas in theta_collections.thetas[0:1]:
        #polar_plot_angles(thetas)
        polar_plot_numbered_angles(thetas)'''

    #polar_plot_numbered_angles(theta_collections.thetas[2])
    '''new_thetas = [thetas for thetas in theta_collections.thetas]
    print(new_thetas[0])
    results = classify_thetas(new_thetas, 5)
    print(results)
    grouped_thetas = group_by_label(new_thetas, results)
    print(grouped_thetas)
    print(to_histogram(results))
    print(results[0])'''

    #print(angle_dist(theta_collections.thetas, partition=10))
    #print(angle_probabilities(theta_collections.thetas, partition=10))
    #print(angle_probabilities(theta_collections.thetas, partition=20))


    #for k_mean_theta in sorted_thetas.thetas[0]:
    #    polar_plot_angles(k_mean_theta)

    '''for polynome, theta in zip(fitted_polynomes.polynomes, fitted_polynomes.theta_vecs):#kmean_thetas[:-3]:
        #plot_fitted_polynome(polynome, theta)
        plot_angles(theta)'''

    '''for theta in theta_collections.thetas[0:10]:
        mat = gdft_matrix(8, theta)
        #print(approximate_matrix(mat, tol=0.5))
        #print(np.absolute(mat-1))
        eigs = np.linalg.eig(mat)
        print(theta)
        plot_eigenvalues(theta)
        #print(approximate_phases(mat, 0.01*np.pi)/np.pi)'''

    #plt.show()
    coeff_8 = np.array([-7.47998864e-03, 1.73916258e-01, -1.61020449e+00, 7.60456544e+00,
                        -1.93127379e+01, 2.45158151e+01, -1.05428434e+01, 2.47251476e-01])

    coeff_4 = np.array([1.04719702, -4.05332898, 2.84656905, 2.63384441])
