import sys
sys.path.append("../src")
import unittest
import numpy as np
from gdft import *
from correlations import *
from optimizer import *

GDFT_MAT = np.array([[1,-1],[-1,-1]], dtype=np.complex128)

class TestOptimizer(unittest.TestCase):

    def setUp(self):
        self.corrs = corr_tensor(dft_matrix(2))
        self.optimizer = Optimizer(2)

    def test_get_correlations(self):
        correlations = self.optimizer.get_correlations(GDFT_MAT)
        self.assertAlmostEqual((0.5, 0.5, 0.5, 1/6), correlations)

    def test_get_random_gdft(self):
        gdft = self.optimizer.get_random_gdft(2)
        #print(gdft)
        #print(gdft*np.transpose(np.conjugate(gdft)))

    def test_generate_results(self):
        '''gdfts = np.zeros((8,8,50), dtype=np.complex128)
        for n in range(50):
            gdfts[:, :, n] = self.optimizer.get_random_gdft(8)
            #print(self.optimizer.get_random_gdft())

        correlations = np.zeros((50, 4), dtype=np.complex128)
        for n in range(50):
            correlations[n, :] = self.optimizer.get_correlations(gdfts[:, :, n])
        print(correlations)
        #print(self.optimizer.generate_results(20))'''
        dft8 = dft_matrix(8)

        correlations = self.optimizer.get_correlations(dft8)
        print(correlations)


    def tearDown(self):
        del self.corrs
        del self.optimizer

if __name__ == '__main__':
    unittest.main()
