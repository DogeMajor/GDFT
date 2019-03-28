from collections import namedtuple
import numpy as np

'''Numpy's fromfunction doesn't work at all in the way it
is described in the np docs. It only calls the fn once and passes
indices to it as a matrix of the shape defined in the args.'''


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

    def correlation_tensor(self):
        N = self._dims[0]
        tensor = np.zeros((N, N, 2 * N - 1), dtype=np.complex128)
        for row in range(N):
            for col in range(row, N):
                for nu in range(2 * N - 1):
                    tensor[row, col, nu] = self._aperiodic_corr_fn(row, col, nu)

        #Set the indices c_col_row : to same as conjugate of c_col_row : with reversed order!!!
        for row in range(N):
            tensor[row, :row, :] = tensor[:row, row, ::-1].conjugate()
        return tensor

    def _aperiodic_corr_fn(self, alpha, beta, pos_mu):
        N = self._dims[0]
        mu = pos_mu - N + 1

        if 0 < mu <= N-1:
            d = np.sum(self._gdft[alpha, :-mu] * self._conj_gdft[beta, mu:])
            return d / N

        d = np.sum(self._gdft[alpha, -mu:] * self._conj_gdft[beta, :N+mu])
        return d/N


#---------correlation analyzer functions --------------

def get_squared_sum(tensor):
    return np.sum(tensor * np.conjugate(tensor))


def get_max_length(tensor):
    abs_tensor = abs(tensor)
    return abs_tensor.max()


def is_auto_corr_mask(shape):
    length = shape[0]
    function = lambda i, j, k: np.logical_and(i == j, k != length - 1)
    ac_mask = np.fromfunction(function, shape, dtype=int)
    return ac_mask


def is_cross_corr_mask(shape):
    cc_mask = np.fromfunction(lambda i, j, k: i != j, shape, dtype=int)
    return cc_mask


Correlations = namedtuple('Correlations',
                          'max_auto_corr avg_auto_corr max_cross_corr avg_cross_corr avg_merit_factor')


class CorrelationAnalyzer(object):

    def __init__(self, dim):
        self._shape = (dim, dim, 2 * dim - 1)
        self._auto_corr_mask = is_auto_corr_mask(self._shape)
        self._cross_corr_mask = is_cross_corr_mask(self._shape)
        self._corr_fns = {"max_auto_corr": self.max_auto_corr,
                          "avg_auto_corr": self.avg_auto_corr,
                          "max_cross_corr": self.max_cross_corr,
                          "avg_cross_corr": self.avg_cross_corr,
                          "avg_merit_factor": self.avg_merit_factor}

    def get_correlations(self, gdft):
        corr_obj = Correlation(gdft)
        corr_tensor = corr_obj.correlation_tensor()
        corrs = {fn_name: corr_fn(corr_tensor) for fn_name, corr_fn in self._corr_fns.items()}
        return Correlations(**corrs)

    def _reduce_corr_tensor(self, mask, calc_fn, c_tensor):
        masked_corr = c_tensor * mask
        return np.real(calc_fn(masked_corr))

    def max_auto_corr(self, c_tensor):
        return self._reduce_corr_tensor(self._auto_corr_mask, get_max_length, c_tensor)

    def avg_auto_corr(self, c_tensor):
        length = c_tensor.shape[0]
        return self._reduce_corr_tensor(self._auto_corr_mask, get_squared_sum, c_tensor) / length

    def max_cross_corr(self, c_tensor):
        return self._reduce_corr_tensor(self._cross_corr_mask, get_max_length, c_tensor)

    def avg_cross_corr(self, c_tensor):
        shape = c_tensor.shape
        length = shape[0] * (shape[0] - 1)
        return self._reduce_corr_tensor(self._cross_corr_mask, get_squared_sum, c_tensor) / length

    def merit_factor(self, alpha, c_tensor):
        shape = c_tensor.shape
        mid_index = int(shape[2] / 2)
        abs_diags = abs(c_tensor[alpha, alpha, :])
        diags_squared = np.power(abs_diags, 2)
        denominator = diags_squared[mid_index]
        nominator = np.sum(diags_squared) - denominator
        return denominator / nominator

    def merit_factors(self, c_tensor):
        shape = c_tensor.shape[0]
        gen = (self.merit_factor(i, c_tensor) for i in range(shape))
        return np.fromiter(gen, np.complex128)

    def avg_merit_factor(self, c_tensor):
        m_factors = self.merit_factors(c_tensor)
        return np.real(m_factors.mean(axis=0))
