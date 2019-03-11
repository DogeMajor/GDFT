import sys
import unittest
from cmath import exp
import numpy as np
from scipy.linalg import expm
sys.path.append("../src")
sys.path.append("src/")
from gdft import dft_matrix, gdft_matrix, two_param_gdft_matrix, g_matrix
from correlations import Correlation, CorrelationAnalyzer
from analyzer import SymmetryAnalyzer


class TestCorrelationAnalyzer(unittest.TestCase):

    def setUp(self):
        self.corrs = Correlation(dft_matrix(2)).correlation_tensor()
        self.analyzer = CorrelationAnalyzer(2)

        #self.analyzer.set_corr_tensor(self.corrs)

    def test_max_auto_correlation(self):
        max_ac = self.analyzer.max_auto_corr(self.corrs)
        self.assertAlmostEqual(max_ac, 0.5)

    def test_avg_auto_correlation(self):
        avg_ac = self.analyzer.avg_auto_corr(self.corrs)
        self.assertAlmostEqual(avg_ac, 0.5)

    def test_max_cross_correlation(self):
        max_cc = self.analyzer.max_cross_corr(self.corrs)
        self.assertAlmostEqual(max_cc, 0.5)
        corr = Correlation(dft_matrix(8)).correlation_tensor()
        analyzer = CorrelationAnalyzer(8)
        #analyzer.set_corr_tensor(corr)
        self.assertAlmostEqual(analyzer.max_cross_corr(corr), 0.327, places=3)

    def test_avg_cross_correlation(self):
        avg_cc = self.analyzer.avg_cross_corr(self.corrs)
        self.assertEqual(avg_cc, 0.5)

    def test_merit_factor(self):
        m_factor = self.analyzer.merit_factor(0, self.corrs)
        self.assertEqual(m_factor, 2)

        m_factor = self.analyzer.merit_factor(1, self.corrs)
        self.assertEqual(m_factor, 2)

    def test_merit_factors(self):
        m_factors = self.analyzer.merit_factors(self.corrs)
        self.assertEqual(list(m_factors), [2, 2])
        merit_avg = self.analyzer.avg_merit_factor(self.corrs)
        self.assertEqual(merit_avg, 2.0)

    '''
    def test_get_correlations(self):
        analyzer = CorrelationAnalyzer(3)
        thetas = np.pi*np.array([0.1, -.2, .3])
        gammas = np.pi*np.array([0, 0, -0.5])
        base_mat = np.ones((3, 3))

        correlations = analyzer.get_correlations(base_mat)
        auto_mask = analyzer._auto_corr_mask
        cross_mask = analyzer._cross_corr_mask
        old_ct = analyzer._corr_tensor
        print(correlations)
        new_matrix = base_mat @ g_matrix(gammas)
        #new_analyzer = CorrelationAnalyzer(3)
        new_correlations = analyzer.get_correlations(new_matrix)
        new_ct = analyzer._corr_tensor
        print(new_correlations)
        print(old_ct.shape)

        small_gdft = two_param_gdft_matrix(2, np.array([1*np.pi, .25*np.pi]), np.array([0.5*np.pi, -0.5*np.pi]))
        print(np.angle(small_gdft)/np.pi)
        print(dft_matrix(2))
        print(self.analyzer.get_correlations(dft_matrix(2)))
        old_ct = self.analyzer._corr_tensor
        print(self.analyzer._corr_tensor.shape)
        print(self.analyzer.get_correlations(small_gdft))
        new_ct = self.analyzer._corr_tensor
        print(self.analyzer._corr_tensor.shape)
        print(np.abs(new_ct) - np.abs(old_ct))'''


    def tearDown(self):
        del self.corrs
        del self.analyzer


if __name__ == '__main__':
    unittest.main()
