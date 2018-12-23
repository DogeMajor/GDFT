import numpy as np
from scipy.linalg import expm

from utils import timer

'''Numpy's fromfunction doesn't work at all in the way it
is described in the np docs. It only calls the fn once and passes
indices to it as a matrix of the shape defined in the args.'''

def pos_acf_mask(shape):
    N = shape[0]
    def pos_corr(i, j, k):
        k = k - N + 1
        return np.logical_and(k > 0, k <= N - 1)
    return np.fromfunction(pos_corr, shape, dtype=int)


class Correlation(object):
    '''Basically a container for the gdft anf conjugated gdft to
    calculate aperiodic correlation tensor.'''

    def __init__(self, gdft):
        self._gdft = gdft
        self._conj_gdft = np.conjugate(self._gdft)
        self._dims = gdft.shape[0], gdft.shape[0], 2*gdft.shape[0]-1

    def _corr_mat(self, mu):

        N = self._dims[0]
        corr_matrix = np.zeros((N, N), dtype=np.complex128)
        for row in range(N):
            for col in range(N):
                corr_matrix[row, col] = self._aperiodic_corr_fn(row, col, mu)
        return corr_matrix

    @timer
    def correlation_tensor(self):
        N = self._dims[0]
        tensor = np.zeros((N, N, 2 * N - 1), dtype=np.complex128)
        for row in range(N):
            for col in range(N):
                for nu in range(2 * N - 1):
                    tensor[row, col, nu] = self._aperiodic_corr_fn(row, col, nu)
        return tensor

    def _aperiodic_corr_fn(self, alpha, beta, pos_mu):
        N = self._dims[0]
        d = 0.0
        mu = pos_mu - N + 1

        if 0 < mu <= N-1:
            d = np.sum(self._gdft[:-mu, alpha] * self._conj_gdft[mu:, beta])
            return d / N

        np.sum(self._gdft[-mu:, alpha] * self._conj_gdft[:N+mu, beta])
        return d/N


#---------correlation analyzers / calculators --------------

def get_squared_sum(c):
    return np.sum(c * np.conjugate(c))

def get_max_length(c):
    abs_c = abs(c)
    return abs_c.max()

def reduce_tensor(c, cond, calc):
    masked_c = c * cond(c)
    return calc(masked_c)


#----auto correlation-------------------------------------

def is_auto_corr(shape):
    length = shape[0]
    ac_mask = np.fromfunction(lambda i, j, k: np.logical_and(i == j, k != length - 1), shape, dtype=int)
    return ac_mask

def is_cross_corr(shape):
    cc_mask = np.fromfunction(lambda i, j, k: i != j, shape, dtype=int)
    return cc_mask

def max_auto_correlation(c_tensor):
    return reduce_tensor(c_tensor, is_auto_corr, get_max_length)

def avg_auto_correlation(c_tensor):
    length = c_tensor.shape[0]
    return reduce_tensor(c_tensor, is_auto_corr, get_squared_sum) / length


#----cross correlation-------------------------------------

def max_cross_correlation(c_tensor):
    return reduce_tensor(c_tensor, is_cross_corr, get_max_length)

def avg_cross_correlation(c_tensor):
    shape = c_tensor.shape
    length = shape[0] * (shape[0] - 1)
    return reduce_tensor(c_tensor, is_cross_corr, get_squared_sum) / length


#------------------------Merit factors--------------------

def merit_factor(c_tensor, alpha):
    shape = c_tensor.shape
    mid_index = int(shape[2]/2)
    abs_diags = abs(c_tensor[alpha, alpha, :])
    diags_squared = np.power(abs_diags, 2)
    denominator = diags_squared[mid_index]
    nominator = np.sum(diags_squared) - denominator
    return denominator/nominator

def merit_factors(c_tensor):
    shape = c_tensor.shape[0]
    gen = (merit_factor(c_tensor, i) for i in range(shape))
    return np.fromiter(gen, np.complex128)

def avg_merit_factor(c_tensor):
    m_factors = merit_factors(c_tensor)
    return m_factors.mean(axis=0)



class CorrelationAnalyzer(object):

    def __init__(self, dim):
        self._shape = (dim, dim, 2 * dim - 1)
        self._corr_tensor = None
        self._auto_corr_mask = is_auto_corr(self._shape)
        self._cross_corr_mask = is_cross_corr(self._shape)

    def set_corr_tensor(self, c_tensor):
        self._corr_tensor = c_tensor

    def _reduce_corr_tensor(self, mask, calc_fn):
        masked_corr = self._corr_tensor * mask
        return calc_fn(masked_corr)

    def max_auto_corr(self):
        return self._reduce_corr_tensor(self._auto_corr_mask, get_max_length)

    @timer
    def avg_auto_corr(self):
        length = self._corr_tensor.shape[0]
        return self._reduce_corr_tensor(self._auto_corr_mask, get_squared_sum) / length

    def max_cross_corr(self):
        return self._reduce_corr_tensor(self._cross_corr_mask, get_max_length)

    def avg_cross_corr(self):
        shape = self._corr_tensor.shape
        length = shape[0] * (shape[0] - 1)
        return self._reduce_corr_tensor(self._cross_corr_mask, get_squared_sum) / length

    def merit_factor(self, alpha):
        shape = self._corr_tensor.shape
        mid_index = int(shape[2] / 2)
        abs_diags = abs(self._corr_tensor[alpha, alpha, :])
        diags_squared = np.power(abs_diags, 2)
        denominator = diags_squared[mid_index]
        nominator = np.sum(diags_squared) - denominator
        return denominator / nominator

    def merit_factors(self):
        shape = self._corr_tensor.shape[0]
        gen = (merit_factor(i) for i in range(shape))
        return np.fromiter(gen, np.complex128)

    def avg_merit_factor(self):
        m_factors = merit_factors()
        return m_factors.mean(axis=0)


