import sys
import unittest
import numpy as np
sys.path.append("../src")
sys.path.append("src/")
from utils import extract_thetas_records
from gdft import gdft_matrix
from correlations import *
from analyzer import PCA, ThetasAnalyzer, SortedThetas
from tools import GDFTTestCase

#------Test data-----------------------------------------------------------------------------

OPTIMIZED_THETAS = extract_thetas_records("data/", "100thetas12-26_1_26.json")

FIRST_THETAS_GROUP = [np.array([2.93467664, 0.3384844, 2.87214115, 1.20613475, 0.32252419,
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

SECOND_THETAS_GROUP = [np.array([0.23263316, 1.06778065, 3.05624654, 2.96119473,
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

SORTED_THETAS = SortedThetas(thetas={0: FIRST_THETAS_GROUP, 1: SECOND_THETAS_GROUP},
                             labels=[FIRST_THETAS_GROUP[0], SECOND_THETAS_GROUP[0]],
                             histogram={}, correlations={})

class TestThetasAnalyzer(GDFTTestCase):

    def setUp(self):
        self.analyzer = ThetasAnalyzer(8)
        self.pca = PCA(8)

    def test_sort_thetas(self):
        sorted_thetas = self.analyzer.sort_thetas(OPTIMIZED_THETAS.thetas, 6)
        self.assertTrue(4 <= len(sorted_thetas.histogram) <= 6)
        self.assertEqual(sorted_thetas.histogram.keys(), sorted_thetas.correlations.keys())
        corr_example = sorted_thetas.correlations[0][0]
        self.assertTrue(corr_example.avg_auto_corr < 0.11)

    def test_solution_spaces(self):
        thetas = {0: OPTIMIZED_THETAS.thetas[0:10], 1: OPTIMIZED_THETAS.thetas[10:20], 2: []}
        first_label = sum(OPTIMIZED_THETAS.thetas[0:10]) / 10
        second_label = sum(OPTIMIZED_THETAS.thetas[10:20]) / 10
        sorted_thetas = SortedThetas(thetas=thetas,
                                     labels=[first_label, second_label],
                                     histogram={}, correlations={})

        sol_spaces = self.analyzer.solution_spaces(sorted_thetas, cutoff_ratio=0.005)
        reduced_covs = self.pca.cov_pca_reductions(sorted_thetas, cutoff_ratio=0.005)

        W0, red_sings = reduced_covs[0]
        reduced_cov = W0.T @ red_sings @ W0 / 10
        self.assertEqual(list(sol_spaces.keys()), [0, 1])
        _, P = sol_spaces[0]['label'], sol_spaces[0]['projection']
        cov0 = self.pca.get_covariance(0, sorted_thetas)
        cov_diff = cov0 - reduced_cov
        self.assertTrue(np.linalg.norm(cov_diff) < 10**-9)

        self.assertAlmostEqualMatrices(P, P.T)
        self.assertAlmostEqualMatrices(P, P.dot(P))

        built_P = W0.T @ W0
        self.assertAlmostEqualMatrices(built_P @ built_P, built_P)
        self.assertAlmostEqualMatrices(built_P.T, built_P)
        self.assertEqual(np.linalg.norm(P - built_P), 0)

        data_matrix = self.pca.to_data_matrix(OPTIMIZED_THETAS.thetas[0:10], subtract_avgs=True)

        self.assertAlmostEqualMatrices(self.pca.get_covariance(0, sorted_thetas),
                                       data_matrix.T @ data_matrix / data_matrix.shape[0])
        self.assertEqual(W0.shape, (4, 8))
        print(red_sings.shape)
        projected_data = data_matrix @ built_P
        data_diff = data_matrix - projected_data
        self.assertTrue(np.linalg.norm(data_diff) < 10**-4)
        vec_diff = data_matrix[0, :] @ P - data_matrix[0, :]
        self.assertTrue(vec_diff.dot(vec_diff) < 10**-10)

        first_label_proj = first_label @ P
        self.assertTrue(np.linalg.norm(first_label - first_label_proj) < 0.0002)

        old_theta = OPTIMIZED_THETAS.thetas[0]
        old_corrs = self.analyzer.get_correlations(gdft_matrix(8, old_theta))
        print(old_corrs.avg_auto_corr)
        direction0 = np.array([2, 7, -12, -6, 2, 11, -9, 2]) @ P
        for n in range(2):
            #new_theta = old_theta + n*np.array([1, 1, 1, 1, 1, 1, 1, 1]) @ P
            new_theta = old_theta + n * 0.01 * direction0
            #print(new_theta)
            new_gdft = gdft_matrix(8, new_theta)
            new_corrs = self.analyzer.get_correlations(new_gdft)
            print(n, new_corrs.avg_auto_corr)

    def test_that_all_sol_spaces_are_similar(self):
        '''Sorted thetas were found out by ThetasAnalyzer'''
        direction1 = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        direction2 = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        u1 = direction1 / np.sqrt(direction1.dot(direction1))
        u2 = direction2 - direction2.dot(u1) * u1
        u2 = u2 / np.sqrt(u2.dot(u2))
        P = np.outer(u1, u1) + np.outer(u2, u2)

        sol_spaces = self.analyzer.solution_spaces(SORTED_THETAS, cutoff_ratio=0.005)
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
                print("Ratio of what part theta_differences do not depend on the 2-dim"
                      "subspace generated by dir1 and dir2 for group {}".format(label_no))

                for theta in all_thetas[1:]:
                    tot_theta_diff = theta - theta_ref
                    theta_diff = tot_theta_diff - P @ tot_theta_diff
                    theta_diff /= np.sqrt(tot_theta_diff.dot(tot_theta_diff))
                    ratio = np.sqrt(theta_diff.dot(theta_diff)) / np.sqrt(tot_theta_diff.dot(tot_theta_diff))
                    print(ratio)
                    self.assertTrue(ratio < 0.02) #I.e. less that 2% of the variation of thetas is
                    # explained by other direction vectors than dir1 and dir2

    def tearDown(self):
        del self.analyzer
        del self.pca

if __name__ == '__main__':
    unittest.main()
