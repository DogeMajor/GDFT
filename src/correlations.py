import numpy as np
from scipy.linalg import expm


class Correlation(np.ndindex):

    def __init__(self, gdft, *shape):
        super().__init__(*shape)
        self._gdft = gdft
        self._conj_gdft = np.conjugate(gdft)
        self._value = (0, 0, 0)
        self._dims = gdft.shape[0], gdft.shape[0], 2*gdft.shape[0]-1
        #self._indices = np.ndindex(self._dims)

    def __iter__(self):
        #self._it = 0
        return self

    def __next__(self):
        print(self._it.iterrange)
        #print(self._it.__repr__)
        #if self._it.has_multi_index(self._value) == True:
        if not self._it.finished:
            current = self._aperiodic_corr_fn(self._value)
            #self.next()
            #print(self._it.multi_index)
            self._value = self._it
            print(self._value)
            return current

        else:
            raise StopIteration

    def _aperiodic_corr_fn(self, index):
        alpha, beta, pos_mu = index[0], index[1], index[2]
        N = self._dims[0]
        d = 0.0
        mu = pos_mu - N + 1
        if abs(mu) >= N:
            return 0.0

        elif mu > 0 and mu <= N-1:
            for nu in range(N-mu):
                d = d + self._gdft[nu, alpha] * self._conj_gdft[nu + mu, beta]
            return d/N

        else:
            for nu in range(N+mu):
                d = d + self._gdft[nu - mu, alpha] * self._conj_gdft[nu, beta]
            return d/N




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
    shape = c_tensor.shape
    m_vecs = np.zeros(shape[0], dtype=np.complex128)
    for alpha in range(shape[0]):
        m_vecs[alpha] = merit_factor(c_tensor, alpha)
    return m_vecs

def avg_merit_factor(c_tensor):
    m_factors = merit_factors(c_tensor)
    return m_factors.mean(axis=0)
