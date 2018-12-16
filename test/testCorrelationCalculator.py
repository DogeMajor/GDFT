import sys
sys.path.append("../src")
import unittest
import numpy as np
from numpy.linalg import inv
from gdft import *
from correlations import *


GDFT_MAT = np.array([[1,-1],[-1,-1]], dtype=np.complex128)


class TestCorrelations(unittest.TestCase):

    def setUp(self):
        self.corrs = corr_tensor(dft_matrix(2))

    def test_max_auto_correlation(self):
        max_ac = max_auto_correlation(self.corrs)
        self.assertAlmostEqual(max_ac, 0.5)

    def test_avg_auto_correlation(self):
        avg_ac = avg_auto_correlation(self.corrs)
        self.assertAlmostEqual(avg_ac, 0.5)

    def test_max_cross_correlation(self):
        max_cc = max_cross_correlation(self.corrs)
        self.assertAlmostEqual(max_cc, 0.5)
        corr = corr_tensor(dft_matrix(8))
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


if __name__ == '__main__':
    unittest.main()
