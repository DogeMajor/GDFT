from cmath import exp
import numpy as np
from scipy.linalg import expm

'''Collection of functions to rapidly generate
gdft, dft and permutation matrices.'''


def dft_matrix(dim):
    W = exp(-1j * 2 * np.pi / dim)
    function = lambda i, j: np.power(W, np.multiply(i, j))
    return np.fromfunction(function, (dim, dim))

def random_unitary_matrix(dim, scaling=10.0):
    gen = scaling * np.random.random((dim, dim))
    return expm(1j * gen)


def g_matrix(phase_shifts): #Solely for testing
    dim = len(phase_shifts)
    g_mat = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(dim):
        g_mat[i, i] = exp(1j*phase_shifts[i])
    return g_mat


def gdft_matrix(dim, thetas):
    dft_mat = dft_matrix(dim)
    gen = (exp(1j * theta) for theta in thetas)
    phase_shifts = np.fromiter(gen, np.complex128)
    return dft_mat * phase_shifts


def two_param_gdft_matrix(dim, gammas, thetas):
    mat = gdft_matrix(dim, thetas)
    gen = (exp(1j * gamma) for gamma in gammas)
    gamma_shifts = np.fromiter(gen, np.complex128)
    return (mat.T * gamma_shifts).T


def permutation_matrix(dim, orderings=None):
    perm = np.zeros((dim, dim))
    for index, order in enumerate(orderings[0:dim]):
        perm[order, index] = 1

    return perm
