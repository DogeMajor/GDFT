from functools import partial
import numpy as np
from scipy.linalg import expm


def dft_matrix(dim):
    W = np.exp(-1j * 2 * np.pi / dim)

    def get_el(row_ind, col_ind):
        return W ** (row_ind * col_ind)
    return np.fromfunction(get_el, (dim, dim), dtype=np.complex128)

def random_unitary_matrix(dim, scaling=10.0):
    gen = scaling * np.random.random((dim, dim))
    return expm(1j * gen)

def g_matrix(phase_shifts):
    diags = expm(1j * np.diag(phase_shifts))
    return np.matrix(diags)

def gdft_matrix(dim, thetas, gammas):
    dft_mat = dft_matrix(dim)
    g1 = g_matrix(thetas)
    g2 = g_matrix(gammas)
    return g1 * dft_mat * g2
