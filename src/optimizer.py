import time
from utils import *
from gdft import *
from correlations import *

np.random.seed(int(time.time()))

class Optimizer(object):

    def __init__(self, dim):
        self._dim = dim
        self._dft = dft_matrix(dim)

    def get_correlations(self, gdft):
        corrs = corr_tensor(gdft)
        max_ac = max_auto_correlation(corrs)
        avg_ac = avg_auto_correlation(corrs)
        max_cc = max_cross_correlation(corrs)
        avg_cc = avg_cross_correlation(corrs)
        return (max_ac, avg_ac, max_cc, avg_cc)

    def get_random_gdft(self, length):
        gammas = np.random.uniform(0, 2 * np.pi, (length))
        thetas = np.random.uniform(0, 2 * np.pi, (length))
        return gdft_matrix(length, thetas, gammas)

    def generate_results(self, length, n):
        gdfts = [self.get_random_gdft(length) for i in range(n)]
        results = [self.get_correlations(gdft) for gdft in gdfts]
        return results
