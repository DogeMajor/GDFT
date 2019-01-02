from functools import partial
import numpy as np
from scipy.linalg import expm
from cmath import exp
from utils import timer


def dft_matrix2(dim):
    W = np.exp(-1j * 2 * np.pi / dim)

    def get_el(row_ind, col_ind):
        return W ** (row_ind * col_ind)
    return np.fromfunction(get_el, (dim, dim), dtype=np.complex128)

def dft_matrix(dim):
    W = exp(-1j * 2 * np.pi / dim)
    fn = lambda i, j: np.power(W, np.multiply(i, j))
    return np.fromfunction(fn, (dim, dim))

def random_unitary_matrix(dim, scaling=10.0):
    gen = scaling * np.random.random((dim, dim))
    return expm(1j * gen)

def g_matrix(phase_shifts):
    dim = len(phase_shifts)
    g_mat = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(dim):
        g_mat[i, i] = exp(1j*phase_shifts[i])
    return g_mat

def gdft_matrix(dim, thetas):
    dft_mat = dft_matrix(dim)
    g1 = g_matrix(thetas)
    return g1.dot(dft_mat.dot(g1))

def non_orthogonal_gdft_matrix(dim, thetas, gammas):
    dft_mat = dft_matrix(dim)
    g1 = g_matrix(thetas)
    g2 = g_matrix(gammas)
    return g1.dot(dft_mat.dot(g2))

@timer
def permutation_matrix(dim, orderings=None):
    perm = np.zeros((dim, dim))
    for index, order in enumerate(orderings):
        perm[index, order] = 1

    return perm
