import sys
import unittest
sys.path.append("../src")
sys.path.append("src/")
from gdft import dft_matrix
from correlations import Correlation, CorrelationAnalyzer


class TestCorrelationAnalyzer(unittest.TestCase):

    def setUp(self):
        self.corrs = Correlation(dft_matrix(2)).correlation_tensor()
        self.analyzer = CorrelationAnalyzer(2)
        self.analyzer.set_corr_tensor(self.corrs)

    def test_max_auto_correlation(self):
        max_ac = self.analyzer.max_auto_corr()
        self.assertAlmostEqual(max_ac, 0.5)

    def test_avg_auto_correlation(self):
        avg_ac = self.analyzer.avg_auto_corr()
        self.assertAlmostEqual(avg_ac, 0.5)
        avg_ac = self.analyzer.avg_auto_corr()

    def test_max_cross_correlation(self):
        max_cc = self.analyzer.max_cross_corr()
        self.assertAlmostEqual(max_cc, 0.5)
        corr = Correlation(dft_matrix(8)).correlation_tensor()
        analyzer = CorrelationAnalyzer(8)
        analyzer.set_corr_tensor(corr)
        self.assertAlmostEqual(analyzer.max_cross_corr(), 0.327, places=3)

    def test_avg_cross_correlation(self):
        avg_cc = self.analyzer.avg_cross_corr()
        self.assertEqual(avg_cc, 0.5)

    def test_merit_factor(self):
        m_factor = self.analyzer.merit_factor(0)
        self.assertEqual(m_factor, 2)

        m_factor = self.analyzer.merit_factor(1)
        self.assertEqual(m_factor, 2)

    def test_merit_factors(self):
        m_factors = self.analyzer.merit_factors()
        self.assertEqual(list(m_factors), [2, 2])
        merit_avg = self.analyzer.avg_merit_factor()
        self.assertEqual(merit_avg, 2.0)

    def tearDown(self):
        del self.corrs
        del self.analyzer


if __name__ == '__main__':
    unittest.main()
