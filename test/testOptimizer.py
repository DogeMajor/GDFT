import sys
sys.path.append("../src")
import unittest
import numpy as np
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
        #self.assertAlmostEqual((0.5, 0.5, 0.5, 1/6), correlations)

    def test_get_random_gdft(self):
        gdft = self.optimizer.get_random_gdft(2)
        #print(gdft)
        #print(gdft*np.transpose(np.conjugate(gdft)))

    def test_generate_results(self):
        '''gdfts = np.zeros((8,8,50), dtype=np.complex128)
        for n in range(50):
            gdfts[:, :, n] = self.optimizer.get_random_gdft(8)
            #print(self.optimizer.get_random_gdft())

        correlations = np.zeros((50, 4), dtype=np.complex128)
        for n in range(50):
            correlations[n, :] = self.optimizer.get_correlations(gdfts[:, :, n])
        print(correlations)
        #print(self.optimizer.generate_results(20))'''
        dft8 = dft_matrix(8)

        correlations = self.optimizer.get_correlations(dft8)
        #print(correlations)

    '''def test_scipys_fmin_bfgs_optimizer(self):
        from scipy.optimize import fmin_bfgs
        def f(x):
            return (x - 3) ** 2, 2 * (x - 3)
        params = fmin_bfgs(lambda x: f(x)[0], 1000, lambda x: f(x)[1])
        self.assertAlmostEqual(params[0], 3)'''

    def test_calc_correlation(self):
        params = [0]*16
        R_ac = self.optimizer._calc_correlation(8, params, avg_auto_correlation)
        self.assertEqual(R_ac, avg_auto_correlation(corr_tensor(dft_matrix(8))))

    def test_optimize_avg_cross_corr(self):
        params0 = self.optimizer._optimize_corr_fn(8, avg_auto_correlation)
        print(params0)
        print(params0[0].shape)
        print(type(params0[0]))
        parameters = self.optimizer._optimize_corr_fn(8, avg_auto_correlation, init_guess=params0[0])
        print(parameters)

    def test_optimize_avg_cross_corr_with_cycles(self):
        params0 = self.optimizer.optimize_corr_fn(8, avg_auto_correlation)
        print(params0)

    def test_constraints(self):
        def constraints(_params):
            return int(any(np.isreal(_params)))

        params = [0] * 16
        print(constraints(params))
        self.assertEqual(int(any(np.iscomplex(params))), 0)


    def tearDown(self):
        del self.corrs
        del self.optimizer

if __name__ == '__main__':
    unittest.main()
