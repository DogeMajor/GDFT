import matplotlib.pyplot as plt
from gdft import *
from utils import *
import numpy as np
from scipy.cluster.vq import kmeans2
from collections import Counter

T = 2*np.pi


plt.grid(True)

def rotate(thetas, deg_angle):
    angle = deg_angle*(np.pi/180)
    return (thetas - angle) % T

def order(thetas):
    return np.sort(thetas)

def _normalize(thetas):
    avg = thetas.mean()
    return thetas - avg

def _to_x(theta):
    return np.cos(theta)

def _to_y(theta):
    return np.sin(theta)

def to_coords(thetas):
    return _to_x(thetas), _to_y(thetas)

def rotate_to_center(thetas, deg_angle):
    angle = deg_angle * (np.pi / 180)
    return thetas - angle



def generate_points(thetas):
    length = thetas.shape[0]
    args = np.array(list(range(length)))
    return args, thetas

def fit_polynome(thetas, grade):
    args, thetas = generate_points(thetas)
    z = np.polyfit(args, thetas, grade)
    f = np.poly1d(z)
    return f

def plot_fitted_polynome(thetas, grade):
    args, thetas = generate_points(thetas)
    pol_fn = fit_polynome(thetas, grade)
    x_new = np.linspace(args[0], args[-1], 50)
    y_new = pol_fn(x_new)
    plt.plot(args, thetas, 'o', x_new, y_new)


def plot_angles(args, thetas):
    plt.plot(args, thetas, 'x')

def polar_plot_angles(thetas):
    x, y = to_coords(thetas)
    plt.plot(x, y, 'o')

def plot_polynome_roots(polynome):
    coeffs = polynome.r
    x, y = coeffs.real, coeffs.imag
    plt.plot(x, y, 'o')


def plot_fn(fn, dim):
    x_new = np.linspace(0, dim-1, 50)
    y_new = fn(x_new)
    plt.plot(x_new, y_new)

def cand_fn(x, dim=8):
    return 0.5*np.pi*(1 + np.sin(np.pi*(x-dim/2)/(dim)))

def cand_fn2(x, dim=8):
    return 0.5*np.pi*(1 + np.sin(np.pi*(x+1-dim/2)/(dim-1)))

def classify_thetas(data, groups):
    return kmeans2(data, groups)

def group_by_label(unsorted_thetas, k_means_results):
    labels = k_means_results[1]
    #label_amount = len(k_means_results[0])
    #sorted_thetas = {label: [] for label in range(label_amount)}
    sorted_thetas = {}
    for ind, theta in enumerate(unsorted_thetas):
        label = labels[ind]
        #sorted_thetas[label].append(theta)
        sorted_thetas.setdefault(label, []).append(theta)

    return sorted_thetas

def to_histogram(k_means_results):
    return Counter(k_means_results[1])


kmean_thetas = [np.array([2.9135797, 0.39698846, 2.63539188, 1.42586124, 0.32580239, 0.41098031, 2.19474127, 3.05086212]),
                np.array([2.92536849, 2.11414487, 0.14960736, 0.26858388, 1.16994527, 2.85369467, 0.33776914, 2.95171189]),
                np.array([0.27610676, 1.03383679, 2.89123087, 2.93963802, 1.90278413, 0.76288471, 2.78617887, 0.4474727]),
                np.array([1.79351973, 2.50482738, 1.67077691, 0.23710056, 2.33149689, 0.42360577, 1.68394482, 1.38386787]),
                np.array([0.25785108, 2.86088575, 0.33405658, 2.00689391, 2.89734625, 3.00541467, 1.02996681, 0.20784047]),
                np.array([4.31449187, 1.12524368, 1.80579287, -0.5236294, 0.56513176, 1.39744013, 0.64624049, 4.16964116])]


if __name__ == "__main__":

    theta_collections = extract_thetas_records("../data/", "results_2018-12-24 23_33.json")
    new_thetas = [thetas for thetas in theta_collections.thetas]
    print(new_thetas[0])
    results = classify_thetas(new_thetas, 6)
    print(results)
    grouped_thetas = group_by_label(new_thetas, results)
    print(grouped_thetas)
    print(to_histogram(results))
    print(results[0])
    #for k_mean_theta in kmean_thetas[:-1]:
    #    polar_plot_angles(k_mean_theta)

    #polar_plot_angles(limited_thetas[6])

    '''for k_mean_theta in kmean_thetas[:-3]:
        plot_fitted_polynome(k_mean_theta, 7)'''

    #plot_fitted_polynome(unordered_thetas[0], 7)
    '''for k_mean_theta in new_thetas:
        polynome = fit_polynome(k_mean_theta, 7)
        plot_polynome_roots(polynome)'''

    #plot_polynome_roots(polynome)

    '''plot_fn(cand_fn2, 8)
    for thetas in unordered_thetas:
        print(thetas)
        #print(16*(1/np.pi)*(thetas-0.5*np.pi))'''
    plt.show()
