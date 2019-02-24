import sys
import unittest
from numpy.linalg import det
import numpy as np
import timeit
sys.path.append("../src")
sys.path.append("src/")

from utils import *
from tools import EqualMatrices, AssertAlmostEqualMatrices
from gdft import *

dft2 = np.array([[1, 1], [1, -1]], dtype=np.complex128)

GDFT_MAT = np.array([[1, -1], [-1, -1]], dtype=np.complex128)

class TestGDFT(unittest.TestCase):

    def setUp(self):
        pass

    def testDFTMatrix(self):
        dft = dft_matrix(2)
        AssertAlmostEqualMatrices(dft, dft2)
        self.assertAlmostEqual(det(dft), -2.0)

    def testRandomUnitaryMatrix(self):
        unitary_mat = random_unitary_matrix(4)
        self.assertAlmostEqual(abs(det(unitary_mat)), 1)
        identity = np.dot(unitary_mat, np.conjugate(unitary_mat))
        AssertAlmostEqualMatrices(np.identity(4), identity)

    def testGMatrix(self):
        g_mat = g_matrix([np.pi/2, -np.pi/2])
        self.assertAlmostEqual(g_mat[0, 0], 1*1j)
        self.assertAlmostEqual(g_mat[0, 1], 0)
        self.assertAlmostEqual(g_mat[1, 1], -1*1j)
        self.assertAlmostEqual(g_mat[1, 0], 0)

    def testGDFTMatrix(self):
        thetas = np.array([-.5 * np.pi, .5 * np.pi])
        gdft_mat = gdft_matrix(2, thetas)
        dft = dft_matrix(2)
        g1 = g_matrix(thetas)
        AssertAlmostEqualMatrices(gdft_mat, np.array([[-1j, -1j], [1j, -1j]], dtype=np.complex128))
        AssertAlmostEqualMatrices(g1.dot(dft), gdft_mat)

    def test_permutation_matrix(self):
        perm = permutation_matrix(2, orderings=[1, 0])
        self.assertTrue(EqualMatrices(perm, np.array([[0, 1], [1, 0]])))
        perm = permutation_matrix(4, orderings=[1, 0, 3, 2, 4])#Works even when someone feeds a too long vector
        self.assertTrue(EqualMatrices(perm, np.array([[0, 1, 0, 0], [1, 0, 0, 0],
                                                      [0, 0, 0, 1], [0, 0, 1, 0]])))

    def tearDown(self):
        pass


def create_small_gdft_matrix():
    thetas = np.ones(8)
    gdft = gdft_matrix(8, thetas)

def create_big_gdft_matrix():
    thetas = np.ones(1000)
    gdft = gdft_matrix(1000, thetas)

class SpeedTests(unittest.TestCase):

    def test_constructing_gdft_mat_with_dim8(self):
        tot_time = timeit.timeit("create_small_gdft_matrix()", setup="from __main__ import create_small_gdft_matrix",
                                 number=10000)
        print("gdft 8x8, 10000 iterations:", tot_time, " s")
        self.assertTrue(tot_time < 0.4)

    def test_constructing_gdft_mat_with_dim1000(self):
        tot_time = timeit.timeit("create_big_gdft_matrix()", setup="from __main__ import create_big_gdft_matrix",
                                 number=1)
        print("gdft 1000x1000, one iteration:", tot_time, "s")
        self.assertTrue(tot_time < 0.6)


if __name__ == '__main__':
    unittest.main()
