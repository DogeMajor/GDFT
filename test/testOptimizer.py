import sys
import unittest
import numpy as np
sys.path.append("../src")
sys.path.append("src/")
from tools import AssertAlmostEqualMatrices
from gdft import gdft_matrix, dft_matrix, two_param_gdft_matrix
from correlations import Correlation, CorrelationAnalyzer
from optimizer import Optimizer


GDFT_MAT = np.array([[1, -1], [-1, -1]], dtype=np.complex128)


class TestOptimizer(unittest.TestCase):

    def setUp(self):
        self.optimizer = Optimizer(8)

    def test_get_correlations(self):
        optimizer = Optimizer(2)
        correlations = optimizer.get_correlations(GDFT_MAT)
        self.assertEqual([correlations.max_auto_corr, correlations.avg_auto_corr,
                          correlations.max_cross_corr, correlations.avg_cross_corr,
                          correlations.avg_merit_factor],
                         [0.5 + 0*1j, 0.5 + 0*1j, 0.5 + 0*1j, 0.5 + 0*1j, 2.0 + 0*1j])

    def test_calc_correlation(self):
        c_tensor = Correlation(dft_matrix(8)).correlation_tensor()
        params = np.zeros(8)
        avg_auto_c = self.optimizer.correlation_functions["avg_auto_corr"]
        average_auto_correlation = self.optimizer._calc_correlation(params, avg_auto_c)

        analyzer = CorrelationAnalyzer(8)

        #analyzer.set_corr_tensor(c_tensor)
        self.assertEqual(average_auto_correlation, analyzer.avg_auto_corr(c_tensor))
        self.assertAlmostEqual(4.375, average_auto_correlation)

    def test_corr_deps_on_params(self):
        thetas = np.array([0.39141802, 0.4717793, 0.49257769, 0.52124477, 2.28552077,
                           3.27810469, 4.8816465, 5.72360472])

        gammas = np.array([0.28036933, 1.06012391, 1.26076182, 2.30176797, 2.75600197,
                           3.21080787, 4.04318979, 4.7630171])

        gdft = gdft_matrix(8, thetas)
        old_correlations = self.optimizer.get_correlations(gdft)
        gammas = np.ones(8)
        new_gdft = two_param_gdft_matrix(8, thetas, gammas)
        new_correlations = self.optimizer.get_correlations(new_gdft)
        for new_corr, old_corr in zip(new_correlations, old_correlations):
            self.assertAlmostEqual(new_corr, old_corr, 2)
        self.assertNotEqual(new_gdft[0, 0], gdft[0, 0])
        AssertAlmostEqualMatrices(np.dot(gdft, np.transpose(np.conjugate(gdft))), 8*np.identity(8))

    def test_optimize_avg_auto_corr(self):

        thetas0, _, _ = self.optimizer._optimize_corr_fn("avg_auto_corr")
        thetas, average_auto_correlation, _ = self.optimizer._optimize_corr_fn("avg_auto_corr",
                                                                               init_guess=thetas0)
        self.assertTrue(0.5 < thetas.mean() < 3.0)
        self.assertTrue(1.0 < thetas.var() < 2.0)
        self.assertTrue(average_auto_correlation < 0.4)

    def test_optimize_avg_cross_corr_with_cycles(self):
        thetas, average_auto_correlation, _ = self.optimizer.optimize_corr_fn("avg_auto_corr",
                                                                              stop_criteria=0.10)
        self.assertTrue(1.0 < thetas.mean() < 2.0)
        self.assertTrue(1.0 < thetas.var() < 2.0)
        self.assertAlmostEqual(average_auto_correlation, 0.0857, places=4)

    def tearDown(self):
        del self.optimizer


if __name__ == '__main__':
    unittest.main()
