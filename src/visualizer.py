import matplotlib.pyplot as plt

import numpy as np
from scipy.cluster.vq import kmeans2
from collections import Counter
from gdft import *
from utils import *
from optimizer import Optimizer
from analyzer import Polynomes, SortedThetas, ThetasAnalyzer

T = 2*np.pi

plt.grid(True)


def _to_x(theta):
    return np.cos(theta)

def _to_y(theta):
    return np.sin(theta)

def to_coords(thetas):
    return _to_x(thetas), _to_y(thetas)

def rotate_to_center(thetas, deg_angle):
    angle = deg_angle * (np.pi / 180)
    return thetas - angle
'''
def generate_points(thetas):
    length = thetas.shape[0]
    args = np.array(list(range(length)))
    return args, thetas

def fit_polynome(thetas, grade):
    args, thetas = generate_points(thetas)
    z = np.polyfit(args, thetas, grade)
    f = np.poly1d(z)
    return f'''

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

def plot_polynome_roots(polynome, max_root_len=None):
    if max_root_len == None:
        max_root_len = len(polynome.c)
    coeffs = polynome.r
    if max(np.abs(coeffs)) < max_root_len :
        x, y = coeffs.real, coeffs.imag
        plt.plot(x, y, 'o')

def plot_fn(fn, dim):
    x_new = np.linspace(0, dim-1, 50)
    y_new = fn(x_new)
    plt.plot(x_new, y_new)

kmean_thetas = [np.array([2.9135797, 0.39698846, 2.63539188, 1.42586124, 0.32580239, 0.41098031, 2.19474127, 3.05086212]),
                np.array([2.92536849, 2.11414487, 0.14960736, 0.26858388, 1.16994527, 2.85369467, 0.33776914, 2.95171189]),
                np.array([0.27610676, 1.03383679, 2.89123087, 2.93963802, 1.90278413, 0.76288471, 2.78617887, 0.4474727]),
                np.array([1.79351973, 2.50482738, 1.67077691, 0.23710056, 2.33149689, 0.42360577, 1.68394482, 1.38386787]),
                np.array([0.25785108, 2.86088575, 0.33405658, 2.00689391, 2.89734625, 3.00541467, 1.02996681, 0.20784047]),
                np.array([4.31449187, 1.12524368, 1.80579287, -0.5236294, 0.56513176, 1.39744013, 0.64624049, 4.16964116])]

new_kmean_thetas = [[0.29941173, 2.89847069, 0.36766799, 2.03652784, 2.92300659,
        3.02710112, 1.04767779, 0.22157399],
       [0.24725148, 1.06828249, 3.04262742, 2.9334619, 2.04191348,
        0.36797856, 2.89371675, 0.28957994],
       [3.01502533, 2.19267569, 0.23495331, 0.35717945, 1.25005117,
        2.92609014, 0.39601341, 3.01691177],
       [1.42282673, 0.95521555, 2.6179977, 1.73874107, 2.26234306,
        2.30620117, 0.74999046, 1.32040115],
       [3.25837994, 1.03165323, 1.28296113, 0.72212853, 0.09730907,
        2.731586, 0.59759348, 3.24538009],
       [3.01110803, 0.40223826, 2.92323331, 1.24455925, 0.3482716,
        0.23436835, 2.20397707, 3.02027523]]

if __name__ == "__main__":
    theta_collections = extract_thetas_records("../data/", "10thetas_16x16__12-27_15_38.json")
    thetas_analyzer = ThetasAnalyzer(16)
    sorted_thetas =
    #theta_collections = extract_thetas_records("../data/", "10thetas_16x16__12-27_11_58.json")
    #theta_collections = thetas = extract_thetas_records("../data/", "10thetas_16x16__12-26_19_4.json")
    #theta_collections = thetas = extract_thetas_records("../data/", "100thetas_4x4__12-26_16_6.json")
    #theta_collections = extract_thetas_records("../data/", "100thetas12-26_1_26.json")
    #theta_collections = extract_thetas_records("../data/", "results_2018-12-24 23_33.json")
    new_thetas = [thetas for thetas in theta_collections.thetas]
    print(new_thetas[0])
    results = classify_thetas(new_thetas, 5)
    print(results)
    grouped_thetas = group_by_label(new_thetas, results)
    print(grouped_thetas)
    print(to_histogram(results))
    print(results[0])

    #for k_mean_theta in new_thetas[:30]:
    #    polar_plot_angles(k_mean_theta)

    #polar_plot_angles(limited_thetas[6])

    for theta in new_thetas:#kmean_thetas[:-3]:
        plot_fitted_polynome(theta, 15)

    #plot_fitted_polynome(unordered_thetas[0], 7)
    '''for k_mean_theta in new_thetas:#grouped_thetas[0]:
        polynome = fit_polynome(k_mean_theta, 15)
        plot_polynome_roots(polynome)'''

    #plot_polynome_roots(polynome)

    '''plot_fn(cand_fn2, 8)
    for thetas in unordered_thetas:
        print(thetas)
        #print(16*(1/np.pi)*(thetas-0.5*np.pi))'''
    plt.show()
