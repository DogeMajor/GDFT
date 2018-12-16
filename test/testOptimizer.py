import sys
sys.path.append("../src")
import unittest
import numpy as np
from tools import *
from gdft import *
from correlations import *
from optimizer import *

GDFT_MAT = np.array([[1,-1],[-1,-1]], dtype=np.complex128)

class TestOptimizer(unittest.TestCase):

    def setUp(self):
        self.corrs = corr_tensor(dft_matrix(2))
        self.optimizer = Optimizer(2)

    def test_get_correlations(self):
        correlations = self.optimizer.get_correlations(GDFT_MAT)
        self.assertAlmostEqual((0.5, 0.5, 0.5, 0.5, 2.0), correlations)

    def test_get_random_gdft(self):
        gdft = self.optimizer.get_random_gdft(8)
        self.assertTrue(abs(np.mean(gdft)) < 0.2)
        #print(gdft*np.transpose(np.conjugate(gdft)))


    def test_calc_correlation(self):
        params = [0]*16
        R_ac = self.optimizer._calc_correlation(8, params, avg_auto_correlation)
        self.assertEqual(R_ac, avg_auto_correlation(corr_tensor(dft_matrix(8))))

    def test_corr_deps_on_params(self):
        thetas = np.array([0.39141802, 0.4717793, 0.49257769, 0.52124477, 2.28552077,
                           3.27810469, 4.8816465, 5.72360472])

        gammas = np.array([0.28036933, 1.06012391, 1.26076182, 2.30176797, 2.75600197,
                           3.21080787, 4.04318979, 4.7630171])

        gdft = gdft_matrix(8, thetas, gammas)
        old_correlations = self.optimizer.get_correlations(gdft)
        gammas = np.ones(8)
        new_gdft = gdft_matrix(8, thetas, thetas)
        new_correlations = self.optimizer.get_correlations(new_gdft)
        for new_corr, old_corr in zip(new_correlations, old_correlations):
            self.assertAlmostEqual(new_corr, old_corr)
        self.assertNotEqual(new_gdft[0, 0], gdft[0, 0])
        self.assertTrue(AlmostEqualMatrices(np.dot(new_gdft, np.conjugate(new_gdft)), 8*np.identity(8)))


    '''def test_optimize_avg_cross_corr(self):
        params0 = self.optimizer._optimize_corr_fn(8, avg_auto_correlation)
        parameters = self.optimizer._optimize_corr_fn(8, avg_auto_correlation, init_guess=params0[0])
        ordered_params = self.optimizer._order_results(parameters)
        print(ordered_params[0])


    def test_optimize_avg_cross_corr_with_cycles(self):
        params0 = self.optimizer.optimize_corr_fn(8, avg_auto_correlation)
        params = self.optimizer._order_results(params0)
        print(params)

    def test_order_results(self):
        parameters = (np.array(range(15, -1, -1)), 9, 're')
        ordered_res = self.optimizer._order_results(parameters)
        self.assertEqual(list(ordered_res[0][8:]), list(range(8)))
        self.assertEqual(list(ordered_res[0][:8]), list(range(8, 16)))
        


    def test_get_optimized_params(self):
        theta_vecs, gamma_vecs = self.optimizer.get_optimized_params(8, avg_auto_correlation, iter_times=10)
        summary = self.optimizer.get_params_summary(theta_vecs, gamma_vecs)
        print(summary)'''


    def tearDown(self):
        del self.corrs
        del self.optimizer

if __name__ == '__main__':
    unittest.main()
