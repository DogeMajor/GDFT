import sys
import unittest
from functools import partial
import numpy as np
from numpy.linalg import inv
sys.path.append("../src")

#from dft import DFT, random_unitary_matrix, aperiodic_corr_fn, aperiodic_correlation_tensor, dft_matrix
from gdft import *


def assertEqualMatrices(matA, matB):
    pass

class TestGDFT(unittest.TestCase):

    def setUp(self):
        pass

    def testDFTMatrix(self):
        dft = dft_matrix(2)
        self.assertEqual

    def testAperiodicCorrelationTensor(self):
        #A_DFT = self.dft_mat
        dft_mat = dft_matrix(2)
        generating_fn = partial(aperiodic_corr_fn, dft_mat)
        self.assertEqual(generating_fn(0,0,1), 0.5)
        corr_tensor = aperiodic_correlation_tensor(dft_mat)
        print(corr_tensor)
    '''
    def testMatrix(self):
        dft_mat = self.dft._dft_matrix
        dft_conj = np.conjugate(dft_mat)
        product = 0.25*dft_mat.dot(dft_conj)
        self.assertAlmostEqual(product[0][0], 1)
        self.assertAlmostEqual(product[0][1], 0)
        self.assertAlmostEqual(product[1][1], 1)

    def testValue(self):
        x = np.array([0, 1.0, 0, 0])
        self.assertEqual(list(self.dft.value(x)), list(self.dft.dft_matrix[:,1]))
        '''
    def tearDown(self):
        pass


if __name__ == '__main__':
    #unittest.main()
    corr = np.zeros((2,2,3), dtype=np.complex128)
    dft_mat = dft_matrix(2)
    for k in range(0, 2):
        for l in range(0, 2):
            for n in range(0, 3):
                res = aperiodic_corr_fn(dft_mat, k, l, n)
                corr[k,l,n] = res

    print(corr[:, :, 0])
    print(corr[:, :, 1])
    print(corr[:, :, 2])

    thetas = np.array([np.pi, -np.pi], dtype=np.complex128)
    gammas = np.array([-0.5*np.pi, 0.5*np.pi], dtype=np.complex128)

    gdft_mat = gdft_matrix(2, [np.pi, -np.pi], [0.5*np.pi, -0.5*np.pi])
    #print(gdft_mat)

    vec = aperiodic_corr_vector(dft_mat, 0, 1)

    print(corr_mat(dft_mat, 0))
    print(corr_mat(dft_mat, 1))
    print(corr_mat(dft_mat, 2))
    correlations = corr_tensor(dft_mat)
    print(correlations.shape)

