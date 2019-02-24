import time
import datetime
from collections import namedtuple
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from utils import timer, datetime_encoder
from dao import DAO
from gdft import gdft_matrix
from correlations import Correlation, CorrelationAnalyzer

np.random.seed(int(time.time()))

'''Optimizes gdft-matrix generating phase shifts with respect to
a chosen correlation measure. Utilizes SciPy's fmin_l_bfgs_b optimizer,
which itself relies on pseudo Newton method of calculating the Hessian matrix.'''

class Optimizer(object):

    def __init__(self, dim):
        self._dim = dim
        self._analyzer = CorrelationAnalyzer(dim)
        self._corr_fns = {"max_auto_corr": self._analyzer.max_auto_corr,
                          "avg_auto_corr": self._analyzer.avg_auto_corr,
                          "max_cross_corr": self._analyzer.max_cross_corr,
                          "avg_cross_corr": self._analyzer.avg_cross_corr,
                          "avg_merit_factor": self._analyzer.avg_merit_factor}

    def get_correlations(self, gdft):
        corr_obj = Correlation(gdft)
        self._analyzer.set_corr_tensor(corr_obj.correlation_tensor())
        Correlations = namedtuple('Correlations', self._corr_fns.keys())
        corrs = {fn_name: corr_fn() for fn_name, corr_fn in self._corr_fns.items()}
        return Correlations(**corrs)

    def _calc_correlation(self, params, corr_fn):
        gdft = gdft_matrix(self._dim, params)
        corr_obj = Correlation(gdft)
        c_tensor = corr_obj.correlation_tensor()
        self._analyzer.set_corr_tensor(c_tensor)
        return corr_fn()

    def _optimize_corr_fn(self, corr_fn_name, init_guess=None):
        if init_guess is None:
            thetas0 = np.random.uniform(0, 2 * np.pi, self._dim)
            init_guess = thetas0

        bnds = tuple((0, np.pi) for n in range(self._dim))
        corr_fn = self._corr_fns[corr_fn_name]

        def output_fn(_params):
            return self._calc_correlation(_params, corr_fn)

        #minimized_params = fmin_bfgs(output_fn, params0, bounds=bnds, constraints=cons)
        minimized_params = fmin_l_bfgs_b(output_fn, init_guess, bounds=bnds, approx_grad=True)
        return minimized_params

    def optimize_corr_fn(self, corr_fn_name, stop_criteria=None, init_guess=None, cycles=10):
        results = self._optimize_corr_fn(corr_fn_name, init_guess)
        for n in range(cycles):
            new_results = self._optimize_corr_fn(corr_fn_name, init_guess)
            print(corr_fn_name, new_results[1])
            if new_results[1] < results[1]:
                results = new_results
            if stop_criteria and results[1] < stop_criteria:
                break
        return results

    def get_params_summary(self, params):
        summary = {}
        thetas = params[0]
        summary['theta_vec'] = thetas
        summary['theta_avg'] = thetas.mean(axis=0)
        summary['theta_var'] = np.var(thetas, axis=0)
        summary['correlation'] = params[1]
        return summary


class Runner(object):

    def __init__(self, dim):
        self._optimizer = Optimizer(dim)

    @timer
    def optimize(self, corr_fn_name, epochs, stop_criteria=None):
        date = datetime_encoder(datetime.datetime.now())
        results = {"info": "Optimizer results for "+corr_fn_name+" at time "+date}
        results["results"] = []
        for n in range(epochs):
            params = self._optimizer.optimize_corr_fn(corr_fn_name, stop_criteria)
            summary = self._optimizer.get_params_summary(params)
            results["results"].append(summary)
        return results

    def save_results(self, file_name, results):
        date_string = datetime_encoder(datetime.datetime.now())
        dao = DAO("../data/")
        dao.write(file_name + date_string + ".json", results)

if __name__ == "__main__":
    runner = Runner(4)
    results = runner.optimize("avg_auto_corr", 30, stop_criteria=0.19)
    runner.save_results("30thetas_4x4__", results)
    #thetas = extract_thetas_records("../data/", "30thetas_16x16__1-1_21_14.json")
    #print(thetas)
