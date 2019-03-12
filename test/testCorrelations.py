import sys
import unittest
import timeit
import numpy as np

sys.path.append("../src")
sys.path.append("src/")
from gdft import dft_matrix
from correlations import Correlation
from temp import *
from tools import EqualMatrices


class TestCorrelation(unittest.TestCase):
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
        self.assertTrue(EqualMatrices(c_tensor[0, 1, :], np.array([3, 5.5, 2])))

    def tearDown(self):
        del self.correlation


SETUP = '''from gdft import gdft_matrix, dft_matrix
from correlations import Correlation
get_corr_tensor = Correlation(dft_matrix(50)).correlation_tensor'''


class SpeedTests(unittest.TestCase):

    def test_how_quickly_correlations_are_computed(self):
        tot_time = timeit.timeit("get_corr_tensor()", setup=SETUP,
                                 number=1)
        print("corr_tensor for dft 50x50:", tot_time, " s")
        self.assertTrue(tot_time < 1.0)

class TestFiniteDifference(unittest.TestCase):

    def setUp(self):
        self.dft = dft_matrix(2)
        self.gdft = gdft_matrix(2, np.array([np.pi, 0.25*np.pi]))
        self.thetas = np.array([np.pi, 0.25*np.pi])

    def test_ct_difference(self):
        corr_tensor = Correlation(self.dft).correlation_tensor()
        old_corr_tensor = Correlation(self.gdft).correlation_tensor()
        thetas = np.array([np.pi, 0.25*np.pi])
        ct_diff0 = ct_difference(thetas, 0, h=0.001)
        self.assertAlmostEqual(ct_diff0[0, 0, 0], -0.35337655+0.35373011j)
        self.assertAlmostEqual(ct_diff0[1, 1, 2], 0.35337655+0.35373011j)
        #print(diff_c_tensor(thetas, 0, h=0.001))

    def test_ct_derivative(self):
        for sigma in range(8):
            thetas8 = np.pi*np.array([1, 0.2, 0.15, -0.5, 2, 0.3, 0.45, -1.5])
            ct_diff = ct_difference(thetas8, sigma, h=0.0000001)
            gdft8 = gdft_matrix(8, thetas8)
            errors = 0
            for row in range(8):
                for col in range(8):
                    for nu in range(15):
                        result_should_be = ct_derivative(gdft8, sigma, row, col, nu)
                        '''if np.abs(ct_diff[row, col, nu] - result_should_be) > 0.01:
                            errors += 1
                            print("False!!!!")
                            print("Theta index: {}".format(sigma))
                            print("indices: {0}, {1}, {2} ".format(row, col, nu))
                            print("Real mu: {}".format(nu-7))
                            print(ct_diff[row, col, nu])
                            print(result_should_be)'''
                        self.assertAlmostEqual(ct_diff[row, col, nu], result_should_be)
        print(errors)

    def test_ct_gradient(self):
        thetas8 = np.pi * np.array([1, 0.2, 0.15, -0.5, 2, 0.3, 0.45, -1.5])
        gdft8 = gdft_matrix(8, thetas8)
        for sigma in range(8):
            ct_diff = ct_difference(thetas8, sigma, h=0.0000001)
            grad_001 = ct_gradient(gdft8, 0, 0, 1)
            self.assertAlmostEqual(grad_001[sigma], ct_diff[0, 0, 1])

    def test_auto_corr_derivative(self):
        thetas8 = np.pi * np.array([1, 0.2, 0.15, -0.5, 2, 0.3, 0.45, -1.5])

        analyzer = CorrelationAnalyzer(8)
        for sigma in range(8):
            ac_diff = corr_difference(analyzer, thetas8, sigma, "avg_auto_corr", h=0.000001)
            gdft8 = gdft_matrix(8, thetas8)
            derivative_Rac0 = auto_corr_derivative(sigma, gdft8)
            self.assertAlmostEqual(derivative_Rac0, ac_diff, 6)

    def test_avg_corr_difference(self):
        thetas8 = np.pi * np.array([1, 0.2, 0.15, -0.5, 2, 0.3, 0.45, -1.5])
        gdft8 = gdft_matrix(8, thetas8)
        avg_corr_difference(thetas8, 0, h=0.00001)

    def test_corr_difference(self):
        thetas8 = np.pi * np.array([1, 0.2, 0.15, -0.5, 2, 0.3, 0.45, -1.5])

        analyzer = CorrelationAnalyzer(8)
        ac_diff = corr_difference(analyzer, thetas8, 0, "avg_auto_corr", h=0.000001)
        print(ac_diff)

    def tearDown(self):
        del self.dft

if __name__ == '__main__':
    unittest.main()
