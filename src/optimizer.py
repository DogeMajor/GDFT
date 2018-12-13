import time
from scipy.optimize import fmin_bfgs, fmin_l_bfgs_b
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

    def _calc_correlation(self, length, params, corr_fn):
        gdft = gdft_matrix(length, params[0:length], params[length:])
        c_tensor = corr_tensor(gdft)
        return corr_fn(c_tensor)

    def _optimize_corr_fn(self, length, corr_fn, init_guess=[]):
        if len(init_guess) == 0:
            gammas0 = np.random.uniform(0, 2 * np.pi, (length))
            thetas0 = np.random.uniform(0, 2 * np.pi, (length))
            init_guess = np.concatenate([gammas0, thetas0])

        bnds = tuple((0, 2*np.pi) for n in range(2*length))

        def cons(_params):
            return float(any(np.iscomplex(_params)))

        def output_fn(_params):
            return self._calc_correlation(length, _params, corr_fn)

        #minimized_params = fmin_bfgs(output_fn, params0, bounds=bnds, constraints=cons)
        minimized_params = fmin_l_bfgs_b(output_fn, init_guess, bounds=bnds, approx_grad=True)
        return minimized_params

    def optimize_corr_fn(self, length, corr_fn, init_guess=[], cycles=10):
        results = self._optimize_corr_fn(length, corr_fn, init_guess)
        for n in range(1, cycles):
            new_results = self._optimize_corr_fn(length, corr_fn, init_guess)
            print("corr", new_results[1])
            if new_results[1] < results[1]:
                results = new_results
        return results





