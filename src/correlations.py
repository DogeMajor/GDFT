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
        tensor[:, :, mu] = corr_mat(matrix, mu)
    return tensor


#---------correlation analyzers / calculators --------------

#----auto correlation-------------------------------------

def scalar_from_tensor(c_tensor, cond_fn, calc_fn):
    scalar = 0
    shape = c_tensor.shape
    for alpha in range(shape[0]):
        for beta in range(shape[1]):
            for mu in range(shape[2]):
                el = c_tensor[alpha, beta, mu]
                if cond_fn(el, alpha, beta, mu, shape[0]):
                    scalar = calc_fn(el, scalar)
    return scalar

def _max_abs_corr(el, scalar):
    if scalar < abs(el):
        return abs(el)
    return scalar

def _is_auto_corr_index(el, alpha, beta, mu, length):
    return alpha == beta and mu != length - 1

def _sum_of_squares(el, scalar):
    return scalar + el * np.conjugate(el)

def max_auto_correlation(c_tensor):
    return scalar_from_tensor(c_tensor, _is_auto_corr_index, _max_abs_corr)

def avg_auto_correlation(c_tensor):
    length = c_tensor.shape[0]
    return scalar_from_tensor(c_tensor, _is_auto_corr_index,_sum_of_squares)/length


#----cross correlation-------------------------------------

def _is_cross_corr_index(el, alpha, beta, mu, length):
    return alpha != beta

def max_cross_correlation(c_tensor):
    return scalar_from_tensor(c_tensor, _is_cross_corr_index, _max_abs_corr)

def avg_cross_correlation(c_tensor):
    shape = c_tensor.shape
    length = shape[0] * (shape[0] - 1)
    return scalar_from_tensor(c_tensor, _is_cross_corr_index, _sum_of_squares) / length
