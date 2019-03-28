import sys
import unittest
import timeit
import numpy as np

sys.path.append("../src")
sys.path.append("src/")
from gdft import dft_matrix
from correlations import Correlation
from tools import EqualMatrices, GDFTTestCase


class TestCorrelation(GDFTTestCase):
    '''OBS! This only tests a symmetric case of matrix A'''

    def setUp(self):
        self.correlation = Correlation(dft_matrix(4))

    def test_aperiodic_corr_fn(self):
        self.assertAlmostEqual(self.correlation._aperiodic_corr_fn(0, 1, -1), 0.0)
        self.assertAlmostEqual(self.correlation._aperiodic_corr_fn(0, 1, 0), 0.25)

    def test_corr_mat(self):
        c_mat = self.correlation._corr_mat(0)
        self.assertEqual(c_mat.shape, (4, 4))
        self.assertEqual(c_mat.dtype, np.complex128)
        self.assertEqual(c_mat[0, 0], 0.25)
        self.assertAlmostEqual(c_mat[3, 3], -0.25 * 1j)

    def test_corr_tensor(self):
        c_tensor = self.correlation.correlation_tensor()
        self.assertEqual(self.correlation._aperiodic_corr_fn(0, 1, 1), c_tensor[0, 1, 1])
        self.assertEqual(self.correlation._aperiodic_corr_fn(1, 0, 2), c_tensor[1, 0, 2])

    def test_asymmetric_c_tensor(self):
        matrix = np.array([[1, 2], [3, 4]])
        corr = Correlation(matrix)
        c_tensor = corr.correlation_tensor()
        self.assertEqualMatrices(c_tensor[0, 1, :], np.array([3, 5.5, 2]))

    def tearDown(self):
        del self.correlation

if __name__ == '__main__':
    unittest.main()
