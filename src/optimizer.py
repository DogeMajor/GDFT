import time
import datetime
from collections import namedtuple
from scipy.optimize import fmin_bfgs, fmin_l_bfgs_b
from utils import *
from gdft import *
from correlations import *

np.random.seed(int(time.time()))

class Optimizer(object):

    def __init__(self, dim):
        self._dim = dim
        self._dft = dft_matrix(dim)
        self._analyzer = CorrelationAnalyzer(dim)
        self._corr_fns = {"max_auto_corr": self._analyzer.max_auto_corr,
                          "avg_auto_corr": self._analyzer.avg_auto_corr,
                          "max_cross_corr": self._analyzer.max_cross_corr,
                          "avg_cross_corr": self._analyzer.avg_cross_corr}

    def get_correlations(self, gdft):
        corr_obj = Correlation(gdft)
        self._analyzer.set_corr_tensor(corr_obj.correlation_tensor())
        max_ac = self._analyzer.max_auto_corr()
        avg_ac = self._analyzer.avg_auto_corr()
        max_cc = self._analyzer.max_cross_corr()
        avg_cc = self._analyzer.avg_cross_corr()
        avg_merit = self._analyzer.avg_merit_factor()
        return (max_ac, avg_ac, max_cc, avg_cc, avg_merit)

    def get_random_gdft(self, length):
        thetas = np.random.uniform(0, 2 * np.pi, (length))
        return gdft_matrix(length, thetas)#With gammas==thetas, gdft remains orthogonal!!!

    def _calc_correlation(self, params, corr_fn):
        gdft = gdft_matrix(self._dim, params)
        corr_obj = Correlation(gdft)
        c_tensor = corr_obj.correlation_tensor()
        self._analyzer.set_corr_tensor(c_tensor)
        return corr_fn()

    def _optimize_corr_fn(self, corr_fn_name, init_guess=[]):
        if len(init_guess) == 0:
            thetas0 = np.random.uniform(0, 2 * np.pi, (self._dim))
            init_guess = thetas0

        bnds = tuple((0, np.pi) for n in range(self._dim))
        corr_fn = self._corr_fns[corr_fn_name]

        def output_fn(_params):
            return self._calc_correlation(_params, corr_fn)

        #minimized_params = fmin_bfgs(output_fn, params0, bounds=bnds, constraints=cons)
        minimized_params = fmin_l_bfgs_b(output_fn, init_guess, bounds=bnds, approx_grad=True)
        return minimized_params

    def optimize_corr_fn(self, corr_fn_name, stop_criteria=None, init_guess=[], cycles=10):
        results = self._optimize_corr_fn(corr_fn_name, init_guess)
        for n in range(1, cycles):
            new_results = self._optimize_corr_fn(corr_fn_name, init_guess)
            print(corr_fn_name, new_results[1])
            if new_results[1] < results[1]:
                results = new_results
            if stop_criteria and results[1] < stop_criteria:
                break
        return results

    def _order_results(self, res):
        params = res[0]
        return np.sort(params), res[1]

    def get_optimized_params(self, corr_fn, iter_times=5):
        '''Iterates optimize_corr_fn several times and returns all the results'''
        params = np.zeros((iter_times, self._dim), dtype=np.complex128)#params, corr_fn_result, iteration
        for n in range(iter_times):
            #params = self._order_results(self.optimize_corr_fn(corr_fn))[0]
            params = self._order_results(self.optimize_corr_fn(corr_fn))[0]

        return params

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
    @save_as_json
    def optimize(self, corr_fn_name, epochs, stop_criteria=None):
        date = datetime_encoder(datetime.datetime.now())
        results = {"info": "Optimizer results for "+corr_fn_name+" at time "+date}
        results["results"] = []
        for n in range(epochs):
            params = self._optimizer.optimize_corr_fn(corr_fn_name, stop_criteria)
            summary = self._optimizer.get_params_summary(params)
            results["results"].append(summary)
        print(results)
        return results



if __name__=="__main__":
    runner = Runner(8)
    runner.optimize("avg_auto_corr", 100, stop_criteria=0.10)
    #thetas = extract_thetas_records("../data/", "results_2018-12-24 23_4.json")
    #print(thetas)



