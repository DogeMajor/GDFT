import sys
import unittest
import timeit
from numpy.linalg import det
import numpy as np
sys.path.append("../src")
sys.path.append("src/")
from tools import EqualMatrices, AssertAlmostEqualMatrices
from gdft import dft_matrix, random_unitary_matrix, g_matrix, gdft_matrix, two_param_gdft_matrix, permutation_matrix


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

    def testGDFTMatrix(self):
        thetas = np.array([-.5 * np.pi, .5 * np.pi])
        gdft_mat = gdft_matrix(2, thetas)
        dft = dft_matrix(2)
        g2 = g_matrix(thetas)
        AssertAlmostEqualMatrices(gdft_mat, np.array([[-1j, 1j], [-1j, -1j]], dtype=np.complex128))
        AssertAlmostEqualMatrices(dft.dot(g2), gdft_mat)

    def test_two_param_GDFTMatrix(self):
        thetas = np.array([-.5 * np.pi, .5 * np.pi])
        gdft_mat = two_param_gdft_matrix(2, -3*thetas, thetas)
        dft = dft_matrix(2)
        g2 = g_matrix(-3*thetas)
        g1 = g_matrix(thetas)
        AssertAlmostEqualMatrices(gdft_mat, np.array([[-1, 1], [1, 1]], dtype=np.complex128))
        AssertAlmostEqualMatrices(g1.dot(dft.dot(g2)), gdft_mat)

    def test_permutation_matrix(self):
        perm = permutation_matrix(2, orderings=[1, 0])
        self.assertTrue(EqualMatrices(perm, np.array([[0, 1], [1, 0]])))
        perm = permutation_matrix(4, orderings=[1, 0, 3, 2, 4])
        self.assertTrue(EqualMatrices(perm, np.array([[0, 1, 0, 0], [1, 0, 0, 0],
                                                      [0, 0, 0, 1], [0, 0, 1, 0]])))

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
