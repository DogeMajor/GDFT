import time
import datetime
import multiprocessing
from collections import namedtuple
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from utils import timer, datetime_encoder
from dao import DAO
from gdft import gdft_matrix
from correlations import Correlation, CorrelationAnalyzer
#np.random.seed(int(time.time()))

'''Optimizes gdft-matrix generating phase shifts with respect to
a chosen correlation measure. Utilizes SciPy's fmin_l_bfgs_b optimizer,
which itself relies on pseudo Newton method of calculating the Hessian matrix.'''

class Optimizer(object):

    def __init__(self, dim):
        self._dim = dim
        self._analyzer = CorrelationAnalyzer(dim)

    @property
    def correlation_functions(self):
        return self._analyzer._corr_fns

    def get_correlations(self, gdft):
        return self._analyzer.get_correlations(gdft)

    def _calc_correlation(self, params, corr_fn):
        gdft = gdft_matrix(self._dim, params)
        corr_obj = Correlation(gdft)
        c_tensor = corr_obj.correlation_tensor()
        self._analyzer.set_corr_tensor(c_tensor)
        return corr_fn()

    def _optimize_corr_fn(self, corr_fn_name, init_guess=None):
        if init_guess is None:
            init_guess = np.pi * np.random.beta(0.5, 0.5, self._dim)
        bnds = tuple((0, np.pi) for n in range(self._dim))
        corr_fn = self.correlation_functions[corr_fn_name]

        def output_fn(_params):
            return self._calc_correlation(_params, corr_fn)

        #minimized_params = fmin_bfgs(output_fn, params0, bounds=bnds, constraints=cons)
        minimized_params = fmin_l_bfgs_b(output_fn, init_guess, bounds=bnds, approx_grad=True)
        return minimized_params

    def optimize_corr_fn(self, corr_fn_name, stop_criteria=None, init_guess=None, cycles=10):
        results = self._optimize_corr_fn(corr_fn_name, init_guess)
        for n in range(cycles):
            new_results = self._optimize_corr_fn(corr_fn_name, init_guess)
            #print(corr_fn_name, new_results[1])
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

    def task(self, index, corr_fn_name, results, stop_criteria):
        params = self._optimizer.optimize_corr_fn(corr_fn_name, stop_criteria)
        results.append(self._optimizer.get_params_summary(params))

    @timer
    def optimize(self, corr_fn_name, epochs, stop_criteria=None):
        date = datetime_encoder(datetime.datetime.now())
        results = {"info": "Optimizer results for "+corr_fn_name+" at time "+date}
        manager = multiprocessing.Manager()
        res = manager.list()
        jobs = []
        func = self._optimizer.optimize_corr_fn
        for epoch in range(epochs):
            proc = multiprocessing.Process(target=self.task, args=(epoch, corr_fn_name, res, stop_criteria))
            jobs.append(proc)
            proc.start()

        for proc in jobs:
            proc.join()
        results["results"] = res
        print(res)
        return results

    def save_results(self, file_name, results):
        date_string = datetime_encoder(datetime.datetime.now())
        dao = DAO("data/")
        dao.write(file_name + date_string + ".json", results)


if __name__ == "__main__":

    runner = Runner(8)
    output = runner.optimize("avg_auto_corr", 5, stop_criteria=0.087)
    #runner.save_results("30thetas_8x8__", results)
    #thetas = extract_thetas_records("../data/", "thetas_16x16__1-1_21_14.json")
    #print(thetas)
