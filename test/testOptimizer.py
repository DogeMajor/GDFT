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

class TestOptimizer(unittest.TestCase):

    def setUp(self):
        #self.corrs = Correlation(dft_matrix(2)).correlation_tensor()
        self.optimizer = Optimizer(8)

    def test_get_correlations(self):
        optimizer = Optimizer(2)
        correlations = optimizer.get_correlations(GDFT_MAT)
        self.assertEqual([correlations.max_auto_corr, correlations.avg_auto_corr,
                          correlations.max_cross_corr, correlations.avg_cross_corr,
                          correlations.avg_merit_factor],
                         [0.5 + 0*1j, 0.5 + 0*1j, 0.5 + 0*1j, 0.5 + 0*1j, 2.0 + 0*1j])

    def test_get_random_gdft(self):
        gdft = self.optimizer.get_random_gdft(8)
        self.assertTrue(abs(np.mean(gdft)) < 0.30)

    def test_calc_correlation(self):
        c_tensor = Correlation(dft_matrix(8)).correlation_tensor()
        c_tensor2 = Correlation(gdft_matrix(8, np.zeros(8))).correlation_tensor()
        params = np.zeros(8)
        avg_auto_c = self.optimizer._corr_fns["avg_auto_corr"]
        R_ac = self.optimizer._calc_correlation(params, avg_auto_c)

        analyzer = CorrelationAnalyzer(8)
        analyzer.set_corr_tensor(c_tensor)
        self.assertEqual(R_ac, analyzer.avg_auto_corr())
        self.assertAlmostEqual(4.375, R_ac)

    def test_corr_deps_on_params(self):
        thetas = np.array([0.39141802, 0.4717793, 0.49257769, 0.52124477, 2.28552077,
                           3.27810469, 4.8816465, 5.72360472])

        gammas = np.array([0.28036933, 1.06012391, 1.26076182, 2.30176797, 2.75600197,
                           3.21080787, 4.04318979, 4.7630171])

        gdft = gdft_matrix(8, thetas)
        old_correlations = self.optimizer.get_correlations(gdft)
        gammas = np.ones(8)
        new_gdft = non_orthogonal_gdft_matrix(8, thetas, gammas)
        new_correlations = self.optimizer.get_correlations(new_gdft)
        for new_corr, old_corr in zip(new_correlations, old_correlations):
            self.assertAlmostEqual(new_corr, old_corr, 2)
        self.assertNotEqual(new_gdft[0, 0], gdft[0, 0])
        self.assertTrue(AlmostEqualMatrices(np.dot(gdft, np.conjugate(gdft)), 8*np.identity(8)))


    def test_optimize_avg_auto_corr(self):

        thetas0, _, _ = self.optimizer._optimize_corr_fn("avg_auto_corr")
        thetas, R_ac, derivative = self.optimizer._optimize_corr_fn("avg_auto_corr", init_guess=thetas0)
        self.assertTrue(0.5 < thetas.mean() < 3.0)
        self.assertTrue(1.0 < thetas.var() < 2.0)
        self.assertTrue(R_ac < 0.4)


    def test_optimize_avg_cross_corr_with_cycles(self):
        thetas, R_ac, _ = self.optimizer.optimize_corr_fn("avg_auto_corr", stop_criteria=0.10)
        self.assertTrue(1.0 < thetas.mean() < 2.0)
        self.assertTrue(1.0 < thetas.var() < 2.0)
        self.assertAlmostEqual(R_ac, 0.0857, places=4)

    def tearDown(self):
        #del self.corrs
        del self.optimizer


if __name__ == '__main__':
    unittest.main()
