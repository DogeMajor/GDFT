import sys
sys.path.append("../src")
sys.path.append("src/")
import unittest
import numpy as np
from gdft import *
from correlations import *


class TestCorrelationAnalyzer(unittest.TestCase):

    def setUp(self):
        self.corrs = Correlation(dft_matrix(2)).correlation_tensor()
        self.analyzer = CorrelationAnalyzer(2)
        self.analyzer.set_corr_tensor(self.corrs)
        print(self.analyzer._auto_corr_mask)
        print(self.analyzer._cross_corr_mask)

    def test_max_auto_correlation(self):
        max_ac = max_auto_correlation(self.corrs)
        self.assertAlmostEqual(max_ac, 0.5)

    '''def test_avg_auto_correlation(self):
        avg_ac = avg_auto_correlation(self.corrs)
        self.assertAlmostEqual(avg_ac, 0.5)
        avg_ac = self.analyzer.avg_auto_corr()

    def test_max_cross_correlation(self):
        max_cc = max_cross_correlation(self.corrs)
        self.assertAlmostEqual(max_cc, 0.5)
        corr = Correlation(dft_matrix(8)).correlation_tensor()
        self.assertAlmostEqual(max_cross_correlation(corr), 0.327, places=3)

    def test_avg_cross_correlation(self):
        avg_cc = avg_cross_correlation(self.corrs)
        self.assertEqual(avg_cc, 0.5)

    def test_merit_factor(self):
        m_factor = merit_factor(self.corrs, 0)
        self.assertEqual(m_factor, 2)

        m_factor = merit_factor(self.corrs, 1)
        self.assertEqual(m_factor, 2)

    def test_merit_factors(self):
        m_factors = merit_factors(self.corrs)
        self.assertEqual(list(m_factors), [2, 2])
        merit_avg = avg_merit_factor(self.corrs)
        self.assertEqual(merit_avg, 2.0)

    def tearDown(self):
        del self.corrs
        del self.analyzer'''


if __name__ == '__main__':
    unittest.main()
