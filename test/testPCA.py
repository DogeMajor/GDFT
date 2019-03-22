import sys
import unittest
import numpy as np
sys.path.append("../src")
sys.path.append("src/")
from analyzer import PCA, SortedThetas
from tools import GDFTTestCase
from utils import extract_thetas_records

#------Test data-----------------------------------------------------------------------------

optimized_thetas = extract_thetas_records("data/", "100thetas12-26_1_26.json")
COV_MAT = np.array([[0.01, 0.02], [0.02, 0.04]])
THETAS = [np.array([1, 2.2]), np.array([1.2, 2.6]), np.array([3, 4])]


class TestPCA(GDFTTestCase):

    def setUp(self):
        self.pca = PCA(8)

    def test_get_covariance(self):
        small_pca = PCA(2)
        sorted_thetas = SortedThetas(thetas={0: [np.array([1, 2.2]),
                                                 np.array([1.2, 2.6])],
                                             1: [np.array([3, 4])]},
                                     labels=np.array([[1.1, 2.25], [3, 4]]),
                                     histogram={0: 2, 1: 1})
        raw_data = np.array([[1, 2.2], [1.2, 2.6]])
        data_matrix = raw_data.mean(axis=0) - raw_data
        cov_mat = small_pca.get_covariance(0, sorted_thetas)
        self.assertAlmostEqualMatrices(cov_mat, COV_MAT)
        self.assertAlmostEqualMatrices(data_matrix.T.dot(data_matrix)/2, COV_MAT)

    def test_get_total_covariance(self):
        small_pca = PCA(2)
        cov_mat = small_pca.get_total_covariance(THETAS)
        self.assertAlmostEqualMatrices(cov_mat,
                                       np.array([[0.8088889, 0.6888889],
                                                 [0.6888889, 0.5955556]]))

    def test_pca_svd_with_data_matrix(self):
        data_matrix = self.pca.to_data_matrix(optimized_thetas.thetas, subtract_avgs=True)
        cov_matrix = (data_matrix.T.dot(data_matrix))/data_matrix.shape[0]
        U, sing_vals, W = self.pca._pca_reduction_svd(data_matrix, cutoff_ratio=0)# no cutoff
        built_data_matrix = U @ sing_vals @ W
        self.assertAlmostEqualMatrices(data_matrix, built_data_matrix)
        self.assertAlmostEqualMatrices(data_matrix.mean(axis=0), data_matrix)
        #'subtract_avgs == True' works as it should
        squared_sing_vals = sing_vals.T @ sing_vals
        self.assertAlmostEqualMatrices(data_matrix, built_data_matrix)
        built_cov_matrix = (W.T @ squared_sing_vals @ W) / data_matrix.shape[0]
        self.assertAlmostEqualMatrices(built_cov_matrix, cov_matrix)
        new_data_matrix = data_matrix @ W.T
        new_cov_matrix = (new_data_matrix.T.dot(new_data_matrix)) / new_data_matrix.shape[0]
        self.assertAlmostEqualMatrices(new_cov_matrix, squared_sing_vals[0:8, :]/100)

    def test_pca_svd_with_cutoff(self):
        data_matrix = self.pca.to_data_matrix(optimized_thetas.thetas, subtract_avgs=True)
        cov_matrix = (data_matrix.T.dot(data_matrix))/data_matrix.shape[0]
        U, sing_vals, W = self.pca._pca_reduction_svd(data_matrix, cutoff_ratio=0.1)# L == 5
        built_data_matrix = U @ sing_vals @ W
        self.assertEqual(sing_vals.shape, (5, 5))
        data_diff = data_matrix[0:10, :] - built_data_matrix[0:10, :]
        self.assertTrue(0.4 < np.linalg.norm(data_diff) < 0.6)
        squared_sing_vals = sing_vals.T @ sing_vals
        built_cov_matrix = (W.T @ squared_sing_vals @ W) / data_matrix.shape[0]
        cov_diff = cov_matrix - built_cov_matrix
        self.assertTrue(0.06 < np.linalg.norm(cov_diff) < 0.07)
        new_data_matrix = data_matrix @ W.T
        new_cov_matrix = (new_data_matrix.T.dot(new_data_matrix)) / new_data_matrix.shape[0]
        self.assertAlmostEqualMatrices(new_cov_matrix, squared_sing_vals/100)

    def test_pca_reduction_svd_with_8dim(self):
        cov_matrix = self.pca.get_total_covariance(optimized_thetas.thetas)
        data_matrix = self.pca.to_data_matrix(optimized_thetas.thetas, subtract_avgs=True)
        U, sing_vals, W = self.pca._pca_reduction_svd(data_matrix, cutoff_ratio=0.23)
        reduced_cov = W.T @ sing_vals.T @ sing_vals @ W / data_matrix.shape[0]
        self.assertEqual(sing_vals.shape, (3, 3))
        self.assertAlmostEqualMatrices(np.diagflat([23.503543, 21.7994326, 5.6939862]), sing_vals)
        self.assertEqual(U.shape, (100, 3))
        self.assertEqual(W.shape, (3, 8))
        self.assertEqual(reduced_cov.shape, (8, 8))
        self.assertAlmostEqual(0.25795731166329816, np.linalg.norm(cov_matrix-reduced_cov))
        self.assertAlmostEqualMatrices(np.identity(3), U.T.dot(U))
        self.assertAlmostEqualMatrices(np.identity(3), W.T.dot(W))
        self.assertTrue(np.abs(cov_matrix - reduced_cov).max() < 0.12)

    def test_get_cov_pca_reductions(self):
        thetas = {0: optimized_thetas.thetas, 1: optimized_thetas.thetas[50:], 2: []}
        sorted_thetas = SortedThetas(thetas=thetas, labels=[], histogram={})
        reduced_covs = self.pca.cov_pca_reductions(sorted_thetas, cutoff_ratio=0.23)
        cov_mat = self.pca.get_total_covariance(optimized_thetas.thetas)
        self.assertEqual(list(reduced_covs.keys()), [0, 1])
        W, sing_mat = reduced_covs[0]
        reduced_cov = W.T @ sing_mat @ W/len(optimized_thetas.thetas)
        self.assertEqual(sing_mat.shape, (3, 3))
        self.assertAlmostEqualMatrices(np.diagflat([5.5241653, 4.7521526, 0.3242148]),
                                       sing_mat / len(optimized_thetas.thetas))
        self.assertEqual(W.shape, (3, 8))
        self.assertEqual(reduced_cov.shape, (8, 8))
        self.assertAlmostEqual(0.257957311663298, np.linalg.norm(cov_mat - reduced_cov))
        self.assertAlmostEqualMatrices(np.identity(3), W.T.dot(W))
        self.assertTrue(np.abs(cov_mat - reduced_cov).max() < 0.12)

    def test_pca_reduction_svd(self):

        data_matrix = self.pca.to_data_matrix(optimized_thetas.thetas, subtract_avgs=True)
        cov_matrix = np.cov(data_matrix.T)
        U, sing_mat, W = self.pca.get_svd(cov_matrix)
        reconstructed_cov_mat = U @ sing_mat @ W
        self.assertAlmostEqualMatrices(reconstructed_cov_mat, cov_matrix)

    def test_distance_of_pca_decomp(self): #They should be zeros if cov_mat is of full rank
        data_matrix = self.pca.to_data_matrix(optimized_thetas.thetas, subtract_avgs=True)
        cov_matrix = data_matrix.T @ data_matrix / 100
        _, sing_mat, _ = self.pca.get_svd(cov_matrix)
        eigen_values, _ = np.linalg.eig(cov_matrix)
        sorted_eigen_values = np.sort(eigen_values, axis=0)
        self.assertAlmostEqualLists(sorted_eigen_values[::-1], sing_mat.diagonal())
        cutoff = 0.3
        distances = [8.45746847732197, 3.854054282310386, 3.854054282310386,
                     2.937969391643371, 0.05979518853865697, 2.9524253032836475e-14]
        dist = 0
        for n in range(6):
            cutoff -= 0.05
            U, sing_vals, W = self.pca._pca_reduction_svd(data_matrix, cutoff_ratio=cutoff)
            reduced_data_matrix = U @ sing_vals @ W
            dist = np.linalg.norm(data_matrix - reduced_data_matrix)
            print(cutoff, dist)
            self.assertAlmostEqual(distances[n], dist)

    def tearDown(self):
        del self.pca

if __name__ == '__main__':
    unittest.main()
