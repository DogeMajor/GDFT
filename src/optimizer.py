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
        c_tensor = corr_tensor(gdft)
        max_ac = max_auto_correlation(c_tensor)
        avg_ac = avg_auto_correlation(c_tensor)
        max_cc = max_cross_correlation(c_tensor)
        avg_cc = avg_cross_correlation(c_tensor)
        avg_merit = avg_merit_factor(c_tensor)
        return (max_ac, avg_ac, max_cc, avg_cc, avg_merit)

    def get_random_gdft(self, length):
        thetas = np.random.uniform(0, 2 * np.pi, (length))
        return gdft_matrix(length, thetas)#With gammas==thetas, gdft remains orthogonal!!!

    def _calc_correlation(self, length, params, corr_fn):
        gdft = gdft_matrix(length, params)
        c_tensor = corr_tensor(gdft)
        return corr_fn(c_tensor)

    def _optimize_corr_fn(self, length, corr_fn, init_guess=[]):
        if len(init_guess) == 0:
            thetas0 = np.random.uniform(0, 2 * np.pi, (length))
            init_guess = thetas0

        bnds = tuple((0, 2*np.pi) for n in range(length))

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


    @show
    def _order_results(self, res):
        params = res[0]
        return np.sort(params), res[1]

    def get_optimized_params(self, length, corr_fn, iter_times=5):
        '''Iterates optimize_corr_fn several times and returns all the results
        i.e. all the phase shifts for both gammas and thetas in ordered form'''
        params = np.zeros((iter_times, length), dtype=np.complex128)#params, corr_fn_result, iteration
        for n in range(iter_times):
            params = self._order_results(self.optimize_corr_fn(length, corr_fn))[0]

        return params

    def get_params_summary(self, thetas):
        summary = {}
        summary['theta_vecs'] = thetas
        summary['theta_avgs'] = thetas.mean(axis=0)
        summary['theta_vars'] = np.var(thetas, axis=0)
        return summary