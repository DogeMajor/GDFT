import sys
sys.path.append("../src")
sys.path.append("src/")
import unittest
from random import shuffle
import numpy as np
from scipy import linalg
from utils import extract_thetas_records, small_els_to, big_els_to, approximate_matrix
from tools import *
from gdft import *
from correlations import *
from analyzer import *
from sequencefinder import SequenceFinder
from tools import GDFTTestCase

#------Test data-----------------------------------------------------------------------------

thetas16x30 = extract_thetas_records("data/", "30thetas_16x16__1-1_21_14.json")
theta8x100 = extract_thetas_records("data/", "100thetas12-26_1_26.json")

normalized_thetas = np.array([-2.98774983e-09, 8.18550897e-01, 2.79042360e+00, 2.67879537e+00,
                              1.78476702e+00, 1.08366030e-01, 2.63164508e+00, 2.50189183e-02])

poly1 = [-7.48008226e-03,  1.73918516e-01, -1.61022589e+00,  7.60466637e+00,
         -1.93129846e+01,  2.45161101e+01, -1.05913241e+01,  3.61686052e-01]

thetas_16gdft = np.array([0.47918196, 3.14159265, 0.37415556, 2.32611506, 0.77481029, 3.08069088,
                          2.36308541, 0.66752458, 2.7953271, 3.07615731, 0.29459556, 0.30038568,
                          0.,        0.,        3.14159265, 3.14159265])

orderings_example = np.array([9, 10, 13, 4, 12, 2, 0, 7, 6, 1, 11, 5, 8, 3, 14, 15])

COV_MAT = np.array([[0.01, 0.02], [0.02, 0.04]])

THETAS = [np.array([1, 2.2]), np.array([1.2, 2.6]), np.array([3, 4])]

thetas_group1 = [np.array([2.93467664, 0.3384844, 2.87214115, 1.20613475, 0.32252419,
                           0.22130658, 2.20358886, 3.03256426]),
                 np.array([2.9696509, 0.32219672, 2.80460713, 1.08736893, 0.15248202,
                           0., 1.93102203, 2.70875109]),
                 np.array([3.14150297, 0.50597848, 3.00032683, 1.29500184, 0.37205216,
                           0.23150659, 2.17446601, 2.96410783]),
                 np.array([2.85346825, 0.24653422, 2.76946155, 1.09272548, 0.19836428,
                           0.08639467, 2.05793914, 2.87617182]),
                 np.array([2.64995925, 0.1094916, 2.69888498, 1.08861413, 0.26072341,
                           0.21521909, 2.25322855, 3.13791887]),
                 np.array([3.11428113, 0.49839706, 3.01236833, 1.32667583, 0.42337081,
                           0.30244392, 2.2650365, 3.07431384]),
                 np.array([2.97236851, 0.3847211, 2.92693977, 1.26949081, 0.39442552,
                           0.30174438, 2.29258057, 3.1300973]),
                 np.array([2.90780283, 0.31334572, 2.84876456, 1.18452025, 0.30265909,
                           0.20317343, 2.18723403, 3.01791636]),
                 np.array([2.73064916, 0.14368992, 2.68659138, 1.02982502, 0.15544191,
                           0.06345344, 2.05497724, 2.89317578]),
                 np.array([2.98336504, 0.33398594, 2.81447009, 1.09527923, 0.15848194,
                           0.00406954, 1.93317593, 2.70895432]),
                 np.array([3.03389687, 0.38015566, 2.85628297, 1.13274342, 0.19158676,
                           0.03281154, 1.95755342, 2.72897963])]

thetas_group2 = [np.array([0.23263316, 1.06778065, 3.05624654, 2.96119473,
                           2.08375977, 0.4239405, 2.96378942, 0.37377238]),
                 np.array([0.12853125, 0.96144711, 2.94767889, 2.85038891,
                           1.97072153, 0.30866741, 2.8462803, 0.25403236]),
                 np.array([0.43171271, 1.20998793, 3.14159265, 2.9896899,
                           2.05536774, 0.33869695, 2.82167874, 0.17477883]),
                 np.array([0.4590644, 1.22344731, 3.14116473, 2.97536152,
                           2.02716572, 0.29658784, 2.76567391, 0.10491548]),
                 np.array([0.18417949, 0.97587018, 2.92086764, 2.7823677,
                           1.86144584, 0.15818738, 2.65457865, 0.02110813]),
                 np.array([0.09761061, 0.91594684, 2.88757434, 2.77571619,
                           1.88147829, 0.2048429, 2.727869, 0.12101726])]

SORTED_THETAS = SortedThetas(thetas={0: thetas_group1, 1: thetas_group2},
                             labels=[thetas_group1[0], thetas_group2[0]],
                             histogram={})

KMEANS_RESULTS = (np.array([[-3.05, -4], [1, 2.05]]), np.array([1, 0]))

UNSORTED_THETAS = [np.array([1, 2.05]), np.array([-3.05, -4])]

class TestClassifier(GDFTTestCase):

    def setUp(self):
        self.classifier = Classifier()

    def test_sort_thetas(self):
        sorted_thetas = self.classifier.sort_thetas(UNSORTED_THETAS, 2)
        self.assertAlmostEqualMatrices(np.sort(sorted_thetas.labels, axis=0), KMEANS_RESULTS[0])
        self.assertEqual(sorted(list(sorted_thetas.histogram.values())), [1, 1])

    def test_group_by_label(self):
        groups = self.classifier.group_by_label(UNSORTED_THETAS, KMEANS_RESULTS)
        vals = list(groups.values())
        self.assertAlmostEqualMatrices(vals[0][0], np.array([1, 2.05]))
        self.assertAlmostEqualMatrices(vals[1][0], np.array([-3.05, -4]))
        self.assertTrue(len(vals[0]) == len(vals[0]) == 1)

    def test_kmeans_to_histogram(self):
        hist = self.classifier._kmeans_to_histogram(KMEANS_RESULTS)
        self.assertEqual(hist, {0: 1, 1: 1})

    def tearDown(self):
        del self.classifier


class TestThetasAnalyzer(GDFTTestCase):

    def setUp(self):
        self.analyzer = ThetasAnalyzer(2)

    def test_get_covariance(self):
        sorted_thetas = SortedThetas(thetas={0: [np.array([1, 2.2]),
                                                 np.array([1.2, 2.6])],
                                             1: [np.array([3, 4])]},
                                     labels=np.array([[1.1, 2.25], [3, 4]]),
                                     histogram={0: 2, 1: 1})
        matA = np.array([[1, 2], [3, 5]])
        raw_data = np.array([[1, 2.2], [1.2, 2.6]])
        data_matrix = raw_data.mean(axis=0) - raw_data
        cov_mat = self.analyzer.get_covariance(0, sorted_thetas)
        self.assertAlmostEqualMatrices(cov_mat, COV_MAT)
        self.assertAlmostEqualMatrices(data_matrix.T.dot(data_matrix)/2, COV_MAT)

    def test_get_total_covariance(self):
        cov_mat = self.analyzer.get_total_covariance(THETAS)
        self.assertAlmostEqualMatrices(cov_mat,
                                       np.array([[0.8088889, 0.6888889],
                                                 [0.6888889, 0.5955556]]))

    def test_pca_svd_with_data_matrix(self):
        analyzer = ThetasAnalyzer(8)
        data_matrix = analyzer.to_data_matrix(theta8x100.thetas, subtract_avgs=True)
        cov_matrix = (data_matrix.T.dot(data_matrix))/data_matrix.shape[0]
        U, sing_vals, W = analyzer._pca_reduction_svd(data_matrix, cutoff_ratio=0)# no cutoff
        built_data_matrix = U @ sing_vals @ W

        self.assertAlmostEqualMatrices(data_matrix, built_data_matrix)
        self.assertAlmostEqualMatrices(data_matrix.mean(axis=0), data_matrix) #'subtract_avgs == True' works as it should
        squared_sing_vals = sing_vals.T @ sing_vals
        self.assertAlmostEqualMatrices(data_matrix, built_data_matrix)
        built_cov_matrix = (W.T @ squared_sing_vals @ W) / data_matrix.shape[0]
        self.assertAlmostEqualMatrices(built_cov_matrix, cov_matrix)
        new_data_matrix = data_matrix @ W.T
        new_cov_matrix = (new_data_matrix.T.dot(new_data_matrix)) / new_data_matrix.shape[0]

        self.assertAlmostEqualMatrices(new_cov_matrix, squared_sing_vals[0:8, :]/100)

    def test_pca_svd_with_cutoff(self):
        analyzer = ThetasAnalyzer(8)
        data_matrix = analyzer.to_data_matrix(theta8x100.thetas, subtract_avgs=True)
        cov_matrix = (data_matrix.T.dot(data_matrix))/data_matrix.shape[0]
        U, sing_vals, W = analyzer._pca_reduction_svd(data_matrix, cutoff_ratio=0.1)# L == 5
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

    def test_numpys_svd(self):
        data_matrix = np.array([[1, 0, 0, 0, 2],
                                [0, 0, 3, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 2, 0, 0, 0]])
        data_matrix = data_matrix - data_matrix.mean(axis=0)
        U, sing_values, W = linalg.svd(data_matrix)

        diags = np.diagflat(sing_values)
        L = diags.shape[0]
        sing_mat = np.block([[diags, np.zeros((L, 5-L))]])
        self.assertAlmostEqualMatrices(U @ sing_mat @ W, data_matrix)
        self.assertAlmostEqualMatrices(U.T @ U, np.eye(5))
        self.assertAlmostEqualMatrices(W @ W.T, np.eye(4))
        self.assertAlmostEqualMatrices(sing_mat, np.array([[2.7751688, 0, 0, 0, 0],
                                                           [0, 2.1213203, 0, 0, 0],
                                                           [0, 0, 1.13949018, 0, 0],
                                                           [0, 0, 0, 0, 0]]))
        cov_sings = sing_mat.T @ sing_mat
        reconstructed_cov = (W.T @ cov_sings @ W)/data_matrix.shape[0]
        self.assertAlmostEqualMatrices((data_matrix.T @ data_matrix)/data_matrix.shape[0], reconstructed_cov)
        analyzer = ThetasAnalyzer(5)
        U_new, sing_mat_new, W_new = analyzer._pca_reduction_svd(data_matrix, cutoff_ratio=0)
        self.assertAlmostEqualMatrices(U_new, U)
        self.assertAlmostEqualMatrices(sing_mat_new, sing_mat)
        self.assertAlmostEqualMatrices(W_new, W)

    def test_pca_reduction_svd_with_8dim(self):
        analyzer = ThetasAnalyzer(8)
        cov_matrix = analyzer.get_total_covariance(theta8x100.thetas)
        data_matrix = analyzer.to_data_matrix(theta8x100.thetas, subtract_avgs=True)
        U, sing_vals, W = analyzer._pca_reduction_svd(data_matrix, cutoff_ratio=0.23)
        reduced_cov = W.T @ sing_vals.T @ sing_vals @ W / data_matrix.shape[0]
        reduced_data_matrix = data_matrix @ W.T
        self.assertEqual(sing_vals.shape, (3, 3))
        self.assertAlmostEqualMatrices(np.diagflat([23.503543, 21.7994326, 5.6939862]), sing_vals)
        self.assertEqual(U.shape, (100, 3))
        self.assertEqual(W.shape, (3, 8))
        self.assertEqual(reduced_cov.shape, (8, 8))
        self.assertAlmostEqual(0.25795731166329816, linalg.norm(cov_matrix-reduced_cov))
        self.assertAlmostEqualMatrices(np.identity(3), U.T.dot(U))
        self.assertAlmostEqualMatrices(np.identity(3), W.T.dot(W))
        self.assertTrue(np.abs(cov_matrix - reduced_cov).max() < 0.12)

    def test_get_cov_pca_reductions(self):
        analyzer = ThetasAnalyzer(8)
        thetas = {0: theta8x100.thetas, 1: theta8x100.thetas[50:], 2: []}
        sorted_thetas = SortedThetas(thetas=thetas, labels=[], histogram={})
        reduced_covs = analyzer.cov_pca_reductions(sorted_thetas, cutoff_ratio=0.23)
        cov_mat = analyzer.get_total_covariance(theta8x100.thetas)
        self.assertEqual(list(reduced_covs.keys()), [0, 1])
        W, sing_mat = reduced_covs[0]
        reduced_cov = W.T @ sing_mat @ W/len(theta8x100.thetas)
        self.assertEqual(sing_mat.shape, (3, 3))
        self.assertAlmostEqualMatrices(np.diagflat([5.5241653, 4.7521526, 0.3242148]), sing_mat/len(theta8x100.thetas))
        self.assertEqual(W.shape, (3, 8))
        self.assertEqual(reduced_cov.shape, (8, 8))
        self.assertAlmostEqual(0.257957311663298, linalg.norm(cov_mat - reduced_cov))
        self.assertAlmostEqualMatrices(np.identity(3), W.T.dot(W))
        self.assertTrue(np.abs(cov_mat - reduced_cov).max() < 0.12)

    def test_solution_spaces(self):
        analyzer = ThetasAnalyzer(8)
        thetas = {0: theta8x100.thetas[0:10], 1: theta8x100.thetas[10:20], 2: []}
        first_label = sum(theta8x100.thetas[0:10])/10
        second_label = sum(theta8x100.thetas[10:20])/10
        sorted_thetas = SortedThetas(thetas=thetas, labels=[first_label, second_label], histogram={})

        sol_spaces = analyzer.solution_spaces(sorted_thetas, cutoff_ratio=0.005)
        reduced_covs = analyzer.cov_pca_reductions(sorted_thetas, cutoff_ratio=0.005)

        W0, red_sings = reduced_covs[0]
        reduced_cov = W0.T @ red_sings @ W0 / 10
        self.assertEqual(list(sol_spaces.keys()), [0, 1])
        theta_mean, P = sol_spaces[0]['label'], sol_spaces[0]['projection']
        cov0 = analyzer.get_covariance(0, sorted_thetas)
        cov_diff = cov0 - reduced_cov
        self.assertTrue(np.linalg.norm(cov_diff) < 10**-9)

        self.assertAlmostEqualMatrices(P, P.T)
        self.assertAlmostEqualMatrices(P, P.dot(P))

        built_P = W0.T @ W0
        self.assertAlmostEqualMatrices(built_P @ built_P, built_P)
        self.assertAlmostEqualMatrices(built_P.T, built_P)
        print(P.shape, built_P.shape)
        self.assertEqual(np.linalg.norm(P - built_P), 0)

        data_matrix = analyzer.to_data_matrix(theta8x100.thetas[0:10], subtract_avgs=True)

        self.assertAlmostEqualMatrices(analyzer.get_covariance(0, sorted_thetas), data_matrix.T @ data_matrix / data_matrix.shape[0])
        self.assertEqual(W0.shape, (4, 8))
        print(red_sings.shape)
        projected_data = data_matrix @ built_P
        data_diff = data_matrix - projected_data
        self.assertTrue(np.linalg.norm(data_diff) < 10**-4)
        vec_diff = data_matrix[0, :] @ P - data_matrix[0, :]
        self.assertTrue(vec_diff.dot(vec_diff) < 10**-10)

        first_label_proj = first_label @ P
        self.assertTrue(np.linalg.norm(first_label - first_label_proj) < 0.0002)

        old_theta = theta8x100.thetas[0]
        old_corrs = analyzer.get_correlations(gdft_matrix(8, old_theta))
        print(old_corrs.avg_auto_corr)
        #print(old_theta)
        min_val = np.min(np.abs(W0[0, :]))
        print(2*W0[0, :] / min_val)
        direction0 = np.array([2, 7, -12, -6, 2, 11, -9, 2]) @ P
        #print(np.array([1, 1, -1, 1, 1, 1, 1, 1]) @ P)
        for n in range(2):
            #new_theta = old_theta + n*np.array([1, 1, 1, 1, 1, 1, 1, 1]) @ P
            new_theta = old_theta + n * 0.01 * direction0
            #print(new_theta)
            new_gdft = gdft_matrix(8, new_theta)
            new_corrs = analyzer.get_correlations(new_gdft)
            print(n, new_corrs.avg_auto_corr)

    def test_that_all_sol_spaces_are_similar(self):
        '''Sorted thetas were found out by ThetasAnalyzer'''
        analyzer = ThetasAnalyzer(8)
        direction1 = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        direction2 = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        u1 = direction1 / np.sqrt(direction1.dot(direction1))
        u2 = direction2 - direction2.dot(u1) * u1
        u2 = u2 / np.sqrt(u2.dot(u2))
        P = np.outer(u1, u1) + np.outer(u2, u2)

        sol_spaces = analyzer.solution_spaces(SORTED_THETAS, cutoff_ratio=0.005)
        P1 = sol_spaces[0]['projection']
        P2 = sol_spaces[1]['projection']
        self.assertEqual(np.linalg.matrix_rank(P1), 2)
        self.assertEqual(np.linalg.matrix_rank(P2), 2)
        self.assertAlmostEqualMatrices(P, P1, decimals=4)
        self.assertAlmostEqualMatrices(P, P2, decimals=4)


        for label_no in sorted(list(SORTED_THETAS.thetas.keys())):
            all_thetas = SORTED_THETAS.thetas[label_no]

            if len(all_thetas) != 0:
                theta_ref = all_thetas[0]
                print("Ratio of what part theta_differences do not depend on the 2-dim "
                      "subspace generated by dir1 and dir2")
                print("for group {}".format(label_no))
                for theta in all_thetas[1:]:
                    tot_theta_diff = theta - theta_ref
                    theta_diff = tot_theta_diff - P @ tot_theta_diff
                    theta_diff /= np.sqrt(tot_theta_diff.dot(tot_theta_diff))
                    ratio = np.sqrt(theta_diff.dot(theta_diff)) / np.sqrt(tot_theta_diff.dot(tot_theta_diff))
                    print(ratio)
                    self.assertTrue(ratio < 0.02) #I.e. less that 2% of the variation of thetas is explained by other
                    #direction vectors than dir1 and dir2


    def test_pca_reduction_svd(self):
        thetas_analyzer = ThetasAnalyzer(8)
        analyzer = thetas_analyzer
        data_matrix = analyzer.to_data_matrix(theta8x100.thetas, subtract_avgs=True)
        cov_matrix = np.cov(data_matrix.T)
        U, sing_mat, W = analyzer.get_svd(cov_matrix)
        reconstructed_cov_mat = U @ sing_mat @ W
        self.assertAlmostEqualMatrices(reconstructed_cov_mat, cov_matrix)

    def test_distance_of_pca_decomp(self): #They should be zeros if cov_mat is of full rank
        analyzer = ThetasAnalyzer(8)
        data_matrix = analyzer.to_data_matrix(theta8x100.thetas, subtract_avgs=True)
        cov_matrix = data_matrix.T @ data_matrix / 100
        u, sing_mat, w = analyzer.get_svd(cov_matrix)
        eigen_values, eigen_vectors = linalg.eig(cov_matrix)
        sorted_eigen_values = np.sort(eigen_values, axis=0)
        self.assertAlmostEqualLists(sorted_eigen_values[::-1], sing_mat.diagonal())
        U_ref, sing_vals_ref, W_ref = analyzer._pca_reduction_svd(data_matrix, cutoff_ratio=0)
        built_data_ref = data_matrix @ W_ref
        cutoff = 0.3
        for n in range(6):
            cutoff -= 0.05
            U, sing_vals, W = analyzer._pca_reduction_svd(data_matrix, cutoff_ratio=cutoff)
            reduced_data_matrix = U @ sing_vals @ W
            print(cutoff, np.linalg.norm(data_matrix - reduced_data_matrix))


    def tearDown(self):
        del self.analyzer

#-----------Help fns for testing how to create a fitting theta-generating function.---

from math import log, sin, asin, tan, sinh, asinh, sqrt, pow, atan

def center(x):
    '''Centers thetas in the interval [0, pi] symmetrically
    with respect to imaginary axis'''
    return x + 0.175584

def func(x):
    return 0.5 * np.pi + atan(center(x))

def transform(seq, fn):
    return [fn(item) for item in seq]

def seq_norm(seq_a, seq_b):
    distances = ((item_b - item_a) ** 2 for item_a, item_b in zip(seq_a, seq_b))
    return sqrt(sum(distances))

def generate_thetas():
    return [0.5 * np.pi + 1.135 * atan(n - 3.63) for n in range(8)]

def generate_thetas_16():
    return [0.5 * np.pi + 1.0 * atan(n - 2*3.75) for n in range(16)]

class TestFindingThetaGeneratingFunction(unittest.TestCase):
    '''Mostly just visual tests to experiment how to build the GDFT from
    the function approximation with permutation matrices.
    Not unit/integration/system tests by any measure!'''

    def setUp(self):
        self.seq_finder = SequenceFinder()
        self.centered_thetas = np.array(transform(normalized_thetas, center))#lulzs

    def test_generating_thetas_with_arctan(self):
        generated_thetas = generate_thetas()
        norm = seq_norm(generated_thetas, sorted(self.centered_thetas))
        print("l^2-norm for the ordered thetas", norm)
        self.assertTrue(norm < 0.3)

    def test_ordering_generated_thetas_with_permutation(self):
        orderings_8 = np.array(self.centered_thetas).argsort()
        self.assertEqual(list(orderings_8), [0, 7, 5, 1, 4, 6, 3, 2])
        perm = permutation_matrix(8, orderings=[0, 7, 5, 1, 4, 6, 3, 2])
        permutated_thetas = perm.dot(generate_thetas())
        self.assertEqual(list(perm.dot(np.sort(self.centered_thetas))), list(self.centered_thetas))
        norm = seq_norm(self.centered_thetas, permutated_thetas)
        print("l^2 norm metric", norm)
        self.assertTrue(norm < 0.3)

    def test_generated_gdft(self):
        permutated_thetas = [0.09304679, 0.93271437, 3.02624859, 2.90047106, 1.97301753, 0.41251557,
                             2.63799845, 0.2003406]
        gdft = gdft_matrix(8, permutated_thetas)
        correlations = ThetasAnalyzer(8).get_correlations(gdft)
        print(correlations)
        self.assertTrue(correlations.avg_auto_corr < 0.12)

    def test_generate_thetas_16(self):
        generated_thetas = generate_thetas_16()
        norm = seq_norm(generated_thetas, sorted(thetas_16gdft))
        print("l^2-norm for the ordered thetas (len = 16)", norm)
        self.assertTrue(norm < 0.7)

    def test_ordering_generated_thetas_16_with_permutation(self):

        orderings = thetas_16gdft.argsort()
        print(orderings)
        self.assertEqual(list(orderings), [12, 13, 10, 11, 2, 0, 7, 4, 3, 6, 8, 9, 5, 1, 14, 15])
        perm = permutation_matrix(16, orderings=orderings)
        permutated_thetas = perm.dot(generate_thetas_16())
        norm = seq_norm(thetas_16gdft, permutated_thetas)
        print("l^2 norm metric for 16 len theta vector", norm)
        self.assertTrue(norm < 0.7)

    def tearDown(self):
        del self.seq_finder
        del self.centered_thetas

'''gdft = gdft_matrix(16, thetas_16gdft)
print(gdft.shape)
correlations = ThetasAnalyzer(16).get_correlations(gdft)
print(correlations.avg_auto_corr)
print(correlations)'''


class TestGDFTBuilder(GDFTTestCase):

    def setUp(self):
        self.builder = GDFTBuilder(8)

    def test_build(self):
        gdft = self.builder.build()
        should_be_identity = gdft.dot(np.conjugate(gdft.T))
        self.assertAlmostEqualMatrices(should_be_identity, 8*np.eye(8))
        correlations = ThetasAnalyzer(8).get_correlations(gdft)
        print(correlations)


    def tearDown(self):
        del self.builder


class TestApproximatingMatrix(GDFTTestCase):

    def setUp(self):
        self.matrix = np.array([[0.09, -0.01], [0.99, 1.15]])
        self.complex_matrix = np.array([[0.02+1j*0.02, -0.01], [0.99-1j*0.001, 1.15+1j*0.04]], dtype=np.complex128)

    def test_small_els_to(self):
        result = small_els_to(self.matrix, replace_val=0, cutoff=0.10)
        self.assertEqualMatrices(result, np.array([[0, 0], [0.99, 1.15]]))
        self.assertAlmostEqualMatrices(np.array([[0, 0], [0.113, 1.15]]), np.array([[0, 0], [0.11, 1.15]]), decimals=1)

    def test_approximate_matrix(self):
        result = approximate_matrix(self.matrix, tol=0.02)
        self.assertAlmostEqualMatrices(result, np.array([[0.09, 0], [1, 1.15]]), decimals=2)

        self.assertAlmostEqualMatrices(approximate_matrix(self.complex_matrix, tol=0.1),
                                  np.array([[0, 0], [1, 1.15+1j*0.04]], dtype=np.complex128), decimals=2)

    def tearDown(self):
        del self.matrix


class TestSymmetryAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = SymmetryAnalyzer(16)
        self.gdft = gdft_matrix(16, thetas_16gdft)

    def test_get_correlations(self):
        corrs = self.analyzer.get_correlations(self.gdft)

    def test_get_similarities(self):
        mat = np.identity(16, np.complex128)
        self.assertEqual(self.analyzer.get_similarities(self.gdft, mat.dot(self.gdft), 0.01),
                         [True, True, True, True, True])
        mat = np.ones((16, 16), np.complex128)
        self.assertEqual(self.analyzer.get_similarities(self.gdft, mat.dot(self.gdft), 0.01),
                         [False, False, False, False, False])

    def test_get_symmetry(self):
        mat = np.identity(16, np.complex128)
        self.assertTrue(self.analyzer.get_symmetry(self.gdft, mat.dot(self.gdft), 0.01))
        mat = np.ones((16, 16), np.complex128)
        self.assertFalse(self.analyzer.get_symmetry(self.gdft, mat.dot(self.gdft), 0.01))


    def tearDown(self):
        del self.analyzer


THETAS4 = np.array([2.23852351, 2.26862803, 0.47525598, 3.14159265])
THETAS8 = np.array([0.15404388, 2.74832147, 0.21274025, 1.87681229, 2.75850199, 2.85781138, 0.87359988, 0.04272007])

class TestSymmetry(GDFTTestCase):

    def setUp(self):
        self.analyzer = SymmetryAnalyzer(8)
        self.gdft = gdft_matrix(8, THETAS8)

    def test_dependency_on_G2(self):
        gammas = np.pi*np.random.rand(8, 1)
        print(gammas)
        new_gdft = two_param_gdft_matrix(8, gammas, THETAS8)  # Correlations are not dependent on gammas!!

        self.assertEqual(self.analyzer.get_similarities(self.gdft, new_gdft, 0.01),
                         [True, True, True, True, True])
        should_be_8_matrix = new_gdft.dot(new_gdft.conjugate().transpose())

        self.assertAlmostEqualMatrices(should_be_8_matrix, 8*np.identity(8))

    def test_similarity_breaking(self):
        new_gdft = gdft_matrix(8, sorted(THETAS8)) #Ordering matters!!
        self.assertEqual(self.analyzer.get_similarities(self.gdft, new_gdft, 0.01),
                         [False, False, False, False, False])

    def test_similarity_preservation(self):
        thetas = THETAS8+0.42 #adding a constant preserves corrs!
        new_gdft = gdft_matrix(8, thetas)
        self.assertEqual(self.analyzer.get_similarities(self.gdft, new_gdft, 0.01),
                         [True, True, True, True, True])

    def test_reversed_order(self):
        orderings = list(range(8))
        perms = permutation_matrix(8, orderings=orderings[::-1])
        thetas = perms.dot(THETAS8)
        new_gdft = gdft_matrix(8, thetas)
        self.assertEqual(self.analyzer.get_similarities(self.gdft, new_gdft, 0.01),
                         [True, True, True, True, True])

    def test_permutations(self):
        for i in range(100):
            orderings = list(range(8))
            shuffle(orderings)
            perms = permutation_matrix(8, orderings=orderings)
            thetas = perms.dot(THETAS8)
            new_gdft = gdft_matrix(8, thetas)
            similarities = self.analyzer.get_similarities(self.gdft, new_gdft, 0.01)
            if sum(map(float, similarities)) > 0:
                print(similarities)
                print(orderings)

    def tearDown(self):
        del self.analyzer


if __name__ == '__main__':
    #pass
    unittest.main()
