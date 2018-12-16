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
        self.dft_mat = dft_matrix(4)

    def testAperiodicCorrelationFn(self):
        self.assertAlmostEqual(aperiodic_corr_fn(self.dft_mat, 0, 1, 7), 0.0)
        self.assertAlmostEqual(aperiodic_corr_fn(self.dft_mat, 0, 1, -1), 0.0)
        self.assertAlmostEqual(aperiodic_corr_fn(self.dft_mat, 0, 1, 0), 0.25)
        self.assertNotAlmostEqual(aperiodic_corr_fn(self.dft_mat, 1, 1, 6), 0.0)

    def testOrigAperiodicCorrelationFn(self):
        dft_mat = dft_matrix(2)
        self.assertEqual(orig_aperiodic_corr_fn(dft_mat, 0, 0, -1), 0.5)
        self.assertEqual(orig_aperiodic_corr_fn(dft_mat, 0, 1, -1), 0.5)
        self.assertEqual(orig_aperiodic_corr_fn(dft_mat, 0, 0, 0), 1.0)
        self.assertEqual(orig_aperiodic_corr_fn(dft_mat, 0, 0, -2), 0.0)
        self.assertEqual(orig_aperiodic_corr_fn(dft_mat, 0, 0, 2), 0.0)
        for n in range(-1,5):
            self.assertEqual(orig_aperiodic_corr_fn(dft_mat, 0, 0, n-1),
                            aperiodic_corr_fn(dft_mat, 0, 0, n))

    def testCorrelationVector(self):
        c_vec = corr_vector(self.dft_mat, 0, 1)
        self.assertEqual(c_vec.shape, (7,))
        self.assertAlmostEqual(c_vec[0], 0.25)
        self.assertAlmostEqual(c_vec[5], -0.25 + 0.25*1j)

    def testCorrelationMatrix(self):
        c_mat = corr_mat(self.dft_mat, 0)
        self.assertEqual(c_mat.shape, (4,4))
        self.assertEqual(c_mat.dtype, np.complex128)
        self.assertEqual(c_mat[0,0], 0.25)
        self.assertAlmostEqual(c_mat[3, 3], 0.25*1j)

    def testCorrelationTensor(self):
        dft_mat = dft_matrix(2)
        corrs = corr_tensor(dft_mat)
        self.assertEqual(aperiodic_corr_fn(dft_mat, 0, 1, 1), corrs[0, 1, 1])
        self.assertEqual(aperiodic_corr_fn(dft_mat, 1, 0, 2), corrs[1, 0, 2])

    def testOutOfPhaseAutoCorrelation(self):
        corrs = corr_tensor(dft_matrix(2))
        max_ac = max_out_of_phase_auto_correlation(corrs)
        self.assertAlmostEqual(max_ac, 0.5)

    def test_avg_auto_correlation(self):
        corrs = corr_tensor(dft_matrix(8))
        avg_ac = avg_auto_correlation(corrs)
        print(avg_ac)


    def test_max_cross_correlation(self):
        corrs = corr_tensor(dft_matrix(2))
        max_cc = max_cross_correlation(corrs)
        self.assertAlmostEqual(max_cc, 1.0)


    def test_avg_cross_correlation(self):
        corrs = corr_tensor(dft_matrix(2))
        avg_cc = avg_cross_correlation(corrs)
        self.assertEqual(avg_cc, 2/3)

    '''def test_avg_auto_correlation(self):
        corrs = corr_tensor(dft_matrix(2))
        avg_ac = avg_auto_correlation(corrs)
        self.assertEqual(avg_ac, 2/3)'''


    def tearDown(self):
        del self.dft_mat


if __name__ == '__main__':
    unittest.main()
