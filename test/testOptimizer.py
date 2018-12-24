import sys
sys.path.append("../src")
sys.path.append("src/")
import unittest
import numpy as np
from tools import *
from gdft import *
from correlations import *
from optimizer import *

GDFT_MAT = np.array([[1, -1], [-1, -1]], dtype=np.complex128)

class TestOptimizer(unittest.TestCase):

    def setUp(self):
        #self.corrs = Correlation(dft_matrix(2)).correlation_tensor()
        self.optimizer = Optimizer(8)

    def test_get_correlations(self):
        optimizer = Optimizer(2)
        correlations = optimizer.get_correlations(GDFT_MAT)
        self.assertAlmostEqual((0.5, 0.5, 0.5, 0.5, 2.0), correlations)

    def test_get_random_gdft(self):
        gdft = self.optimizer.get_random_gdft(8)
        self.assertTrue(abs(np.mean(gdft)) < 0.25)

    def test_calc_correlation(self):
        c_tensor = Correlation(dft_matrix(8)).correlation_tensor()
        c_tensor2 = Correlation(gdft_matrix(8, np.zeros(8))).correlation_tensor()
        params = np.zeros(8)
        avg_auto_c = self.optimizer._corr_fns["avg_auto_corr"]
        R_ac = self.optimizer._calc_correlation(8, params, avg_auto_c)

        analyzer = CorrelationAnalyzer(8)
        analyzer.set_corr_tensor(c_tensor)
        self.assertEqual(R_ac, analyzer.avg_auto_corr())

    def test_corr_deps_on_params(self):
        thetas = np.array([0.39141802, 0.4717793, 0.49257769, 0.52124477, 2.28552077,
                           3.27810469, 4.8816465, 5.72360472])

        gammas = np.array([0.28036933, 1.06012391, 1.26076182, 2.30176797, 2.75600197,
                           3.21080787, 4.04318979, 4.7630171])

        gdft = gdft_matrix(8, thetas)
        old_correlations = self.optimizer.get_correlations(gdft)
        gammas = np.ones(8)
        new_gdft = non_orthogonal_gdft_matrix(8, thetas, gammas)
        new_correlations = self.optimizer.get_correlations(new_gdft)
        for new_corr, old_corr in zip(new_correlations, old_correlations):
            self.assertAlmostEqual(new_corr, old_corr, 2)
        self.assertNotEqual(new_gdft[0, 0], gdft[0, 0])
        self.assertTrue(AlmostEqualMatrices(np.dot(gdft, np.conjugate(gdft)), 8*np.identity(8)))


    def test_optimize_avg_auto_corr(self):

        params0 = self.optimizer._optimize_corr_fn(8, "avg_auto_corr")
        parameters = self.optimizer._optimize_corr_fn(8, "avg_auto_corr", init_guess=params0[0])
        ordered_params = self.optimizer._order_results(parameters)



    def test_optimize_avg_cross_corr_with_cycles(self):
        params0 = self.optimizer.optimize_corr_fn(8, "avg_auto_corr")
        params = self.optimizer._order_results(params0)
        print(params0)

    '''def test_if_the_order_of_params_changes_corrs(self):
        thetas = np.array([0.05249126, 2.72382596, 0.26529619, 2.00643631, 2.96519007,
                           3.14156252, 1.23441539, 0.48058796])
        gdft = gdft_matrix(8,thetas)
        corrs = self.optimizer.get_correlations(gdft)
        #print(corrs)
        def swap_els(vec, i, j):
            new_vec = np.array(vec)
            new_vec[i], new_vec[j] = vec[j], vec[i]
            return new_vec

        for i in range(8):
            new_thetas = swap_els(thetas, 7, i)
            #print(i, new_thetas)
            new_gdft = gdft_matrix(8, new_thetas)
            new_corrs = self.optimizer.get_correlations(new_gdft)
            print(i, new_corrs)

    
    def test_order_results(self):
        parameters = (np.array(range(15, -1, -1)), 9, 're')
        ordered_res = self.optimizer._order_results(parameters)
        self.assertEqual(list(ordered_res[0]), list(range(16)))



    def test_get_optimized_params(self):
        thetas = self.optimizer.get_optimized_params(8, avg_auto_correlation, iter_times=10)
        summary = self.optimizer.get_params_summary(thetas)
        print(summary)'''


    def tearDown(self):
        #del self.corrs
        del self.optimizer




if __name__ == '__main__':
    unittest.main()
    '''thetas = np.array([0.15759434, 0.27662528, 0.33310666, 0.98958436, 1.99608701,
                       2.86979638, 2.87667988, 2.97489775], dtype=np.complex128)
    verify_optimal_thetas(thetas)
    optimizer = Optimizer(8)
    gdft = gdft_matrix(8, thetas)
    print(optimizer.get_correlations(gdft))
    params = np.array([0.84401301, 2.58966888, 3.14159265, 0.,         3.14159265, 1.27085015,
     1.35837713, 0.22960154])
     '''
