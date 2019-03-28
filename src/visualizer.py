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
    records = [(theta, corr) for theta, corr
               in zip(thetas_collections.thetas, theta_collections.correlations)]
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


def center_thetas(all_thetas, index=0, epsilon=1e-10):
    # Epsilon is needed to keep the data matrix pos. definite!!!
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
    theta_collections = extract_thetas_records("../data/", "R_ac_100thetas_16x16__3-21_14_11.json")
    centered_thetas = center_thetas(theta_collections.thetas)
    #theta_collections = extract_thetas_records("../data/", "results_2018-12-24 23_33.json")
    sorted_thetas = thetas_analyzer.sort_thetas(centered_thetas, 6)
    corr_analyzer = CorrelationAnalyzer(16)

    direction0 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    direction1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

    u0 = direction0 / np.sqrt(direction0.dot(direction0))
    u1 = direction1 - direction1.dot(u0) * u0
    u1 = u1 / np.sqrt(u1.dot(u1))
    P = np.outer(u0, u0) + np.outer(u1, u1)
    data_matrix = thetas_analyzer.to_data_matrix(theta_collections.thetas[0:10], subtract_avgs=True)
