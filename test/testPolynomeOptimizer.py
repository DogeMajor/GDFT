import sys
sys.path.append("../src")
sys.path.append("src/")
import unittest
import numpy as np
from tools import *
from gdft import *
from correlations import *
from optimizer import *

GDFT_MAT = np.array([[1, -1], [-1, -1]], dtype=np.complex128)


class TestPolynomeOptimizer(unittest.TestCase):

    def setUp(self):
        self.corrs = Correlation(dft_matrix(2)).correlation_tensor()
        self.optimizer = PolynomeOptimizer(8)

    def test_get_correlations(self):
        correlations = self.optimizer.get_correlations(GDFT_MAT)
        self.assertAlmostEqual((0.5, 0.5, 0.5, 0.5, 2.0), correlations)

    def test_coeffs_to_thetas(self):
        #coeffs = [0] * 8
        coeffs = np.ones(8)
        thetas = self.optimizer._coeffs_to_thetas(coeffs)
        #print(thetas)

    def test_get_poly_bounds(self):
        bounds = self.optimizer._get_poly_bounds(8)
        print(bounds)
        self.assertEqual(bounds[7][0], -np.pi)
        self.assertEqual(bounds[0][0], -np.pi*8**-8)

        thetas = self.optimizer._coeffs_to_thetas(coeffs)

    def test_calc_correlation(self):
        params = [0] * 8
        R_ac = self.optimizer._calc_correlation(7, params, avg_auto_correlation)
        c_tensor = Correlation(dft_matrix(8)).correlation_tensor()
        self.assertEqual(R_ac, avg_auto_correlation(c_tensor))

    def test_optimize_avg_auto_corr(self):
        params0 = self.optimizer._optimize_corr_fn(8, avg_auto_correlation)
        parameters = self.optimizer._optimize_corr_fn(8, avg_auto_correlation, init_guess=params0[0])
        ordered_params = self.optimizer._order_results(parameters)
        print(parameters)
        thetas = self.optimizer._coeffs_to_thetas(parameters[0])
        print(thetas)



    '''
    def test_optimize_avg_cross_corr_with_cycles(self):
        params0 = self.optimizer.optimize_corr_fn(8, avg_auto_correlation)
        params = self.optimizer._order_results(params0)
        print(params0)

    def test_if_the_order_of_params_changes_corrs(self):
        thetas = np.array([0.05249126, 2.72382596, 0.26529619, 2.00643631, 2.96519007,
                           3.14156252, 1.23441539, 0.48058796])
        gdft = gdft_matrix(8,thetas)
        corrs = self.optimizer.get_correlations(gdft)
        #print(corrs)
        def swap_els(vec, i, j):
            new_vec = np.array(vec)
            new_vec[i], new_vec[j] = vec[j], vec[i]
            return new_vec

        for i in range(8):
            new_thetas = swap_els(thetas, 7, i)
            #print(i, new_thetas)
            new_gdft = gdft_matrix(8, new_thetas)
            new_corrs = self.optimizer.get_correlations(new_gdft)
            print(i, new_corrs)


    def test_order_results(self):
        parameters = (np.array(range(15, -1, -1)), 9, 're')
        ordered_res = self.optimizer._order_results(parameters)
        self.assertEqual(list(ordered_res[0]), list(range(16)))



    def test_get_optimized_params(self):
        thetas = self.optimizer.get_optimized_params(8, avg_auto_correlation, iter_times=10)
        summary = self.optimizer.get_params_summary(thetas)
        print(summary)'''

    def tearDown(self):
        del self.corrs
        del self.optimizer

if __name__ == "__main__":
    unittest.main()