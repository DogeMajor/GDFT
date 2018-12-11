from functools import partial
import numpy as np
from scipy.linalg import expm


def aperiodic_corr_fn(matrix, alpha, beta, pos_mu):
    N = matrix.shape[0]
    d = 0.0
    conj_matrix = np.conjugate(matrix)
    mu = pos_mu - N + 1
    if abs(mu) >= N:
        return 0.0

    elif mu > 0 and mu <= N-1:
        for nu in range(N-mu):
            d = d + matrix[nu, alpha] * conj_matrix[nu + mu, beta]
        return d/N

    else:
        for nu in range(N+mu):
            d = d + matrix[nu - mu, alpha] * conj_matrix[nu, beta]
        return d/N

def orig_aperiodic_corr_fn(matrix, alpha, beta, mu_real):
    # 1-N <= mu <= N-1
    mu_positive = mu_real + matrix.shape[0] - 1
    return aperiodic_corr_fn(matrix, alpha, beta, mu_positive)


def corr_vector(matrix, alpha, beta):
    N = matrix.shape[0]
    vec = [aperiodic_corr_fn(matrix, alpha, beta, mu) for mu in range(2*N-1)]
    return np.array(vec)

def corr_mat(matrix, mu):
    N = matrix.shape[0]
    corr_matrix = np.zeros((N, N), dtype=np.complex128)
    for row in range(N):
        for col in range(N):
            corr_matrix[row, col] = aperiodic_corr_fn(matrix, row, col, mu)
    return corr_matrix

def corr_tensor(matrix):
    N = matrix.shape[0]
    tensor = np.zeros((N, N, 2*N-1), dtype=np.complex128)
    for mu in range(2*N-1):
        tensor[:,:,mu] = corr_mat(matrix, mu)
    return tensor
