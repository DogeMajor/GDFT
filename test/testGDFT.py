import sys
import unittest
from numpy.linalg import det
import numpy as np
sys.path.append("../src")
from tools import EqualMatrices, AlmostEqualMatrices
from gdft import *

dft2 = np.array([[1,1],[1,-1]], dtype=np.complex128)

class TestGDFT(unittest.TestCase):

    def setUp(self):
        pass

    def testDFTMatrix(self):
        dft = dft_matrix(2)
        self.assertTrue(AlmostEqualMatrices(dft, dft2))
        self.assertAlmostEqual(det(dft), -2.0)

    def testRandomUnitaryMatrix(self):
        unitary_mat = random_unitary_matrix(4)
        self.assertAlmostEqual(abs(det(unitary_mat)), 1)
        identity = np.dot(unitary_mat, np.conjugate(unitary_mat))
        self.assertTrue(AlmostEqualMatrices(np.identity(4), identity))

    def testGMatrix(self):
        g_mat = g_matrix([np.pi/2, -np.pi/2])
        self.assertAlmostEqual(g_mat[0, 0], 1*1j)
        self.assertAlmostEqual(g_mat[0, 1], 0)
        self.assertAlmostEqual(g_mat[1, 1], -1*1j)
        self.assertAlmostEqual(g_mat[1, 0], 0)

    def testGDFTMatrix(self):
        thetas = [-0.5 * np.pi, 0.5 * np.pi]
        gammas = [0.5 * np.pi, -0.5 * np.pi]
        gdft_mat = gdft_matrix(2, thetas, gammas)
        dft = dft_matrix(2)
        g1 = g_matrix(thetas)
        g2 = g_matrix(gammas)
        self.assertTrue(AlmostEqualMatrices(np.array([[1, -1], [-1, -1]], dtype=np.complex128), gdft_mat))

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
