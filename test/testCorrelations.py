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
        self.assertAlmostEqual(c_vec[5], -0.25 - 0.25*1j)

    def testCorrelationMatrix(self):
        c_matr = corr_mat(self.dft_mat, 0)
        self.assertEqual(c_matr.shape, (4,4))
        self.assertEqual(c_matr.dtype, np.complex128)
        self.assertEqual(c_matr[0,0], 0.25)
        self.assertAlmostEqual(c_matr[3, 3], -0.25*1j)

    def testCorrelationTensor(self):
        dft_mat = dft_matrix(4)
        corrs = corr_tensor(dft_mat)
        self.assertEqual(aperiodic_corr_fn(dft_mat, 0, 1, 1), corrs[0, 1, 1])
        self.assertEqual(aperiodic_corr_fn(dft_mat, 1, 0, 2), corrs[1, 0, 2])


    def tearDown(self):
        del self.dft_mat


class TestCorrelation(unittest.TestCase):

    def setUp(self):
        self.correlation = Correlation(dft_matrix(4))

    def test_aperiodic_corr_fn(self):
        self.assertAlmostEqual(self.correlation._aperiodic_corr_fn(0, 1, 7), 0.0)
        self.assertAlmostEqual(self.correlation._aperiodic_corr_fn(0, 1, -1), 0.0)
        self.assertAlmostEqual(self.correlation._aperiodic_corr_fn(0, 1, 0), 0.25)
        self.assertNotAlmostEqual(self.correlation._aperiodic_corr_fn(1, 1, 6), 0.0)


    def test_corr_mat(self):
        #print((0,0,0) in range(2,2,4))
        c_mat = self.correlation._corr_mat(0)
        #c_mat = corr_mat(self.dft_mat, 0)
        self.assertEqual(c_mat.shape, (4, 4))
        self.assertEqual(c_mat.dtype, np.complex128)
        self.assertEqual(c_mat[0, 0], 0.25)
        self.assertAlmostEqual(c_mat[3, 3], -0.25 * 1j)

    def test_corr_tensor(self):
        #print((0,0,0) in range(2,2,4))
        c_tensor = self.correlation.correlation_tensor()
        self.assertEqual(self.correlation._aperiodic_corr_fn(0, 1, 1), c_tensor[0, 1, 1])
        self.assertEqual(self.correlation._aperiodic_corr_fn(1, 0, 2), c_tensor[1, 0, 2])

    def test_speed_diffs(self):
        from functools import partial
        import timeit
        dft_mat = dft_matrix(16)
        c = Correlation(dft_mat)
        fn1 = c.correlation_tensor
        print(timeit.timeit(fn1, number=100))

        fn2 = partial(corr_tensor, dft_mat)
        print(timeit.timeit(fn2, number=100))

    def tearDown(self):
        del self.correlation

if __name__ == '__main__':
    unittest.main()
