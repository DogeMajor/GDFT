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

class TestThetasAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = ThetasAnalyzer(2)

    def test_sort_thetas(self):
        sorted_thetas = self.analyzer.sort_thetas([[1, 2.05], [-3.05, -4]], 2)
        AssertAlmostEqualMatrices(np.sort(sorted_thetas.labels, axis=0), np.array([[-3.05, -4.], [1, 2.05]]))
        self.assertEqual(sorted(list(sorted_thetas.histogram.values())), [1, 1])

    def test_get_covariance(self):
        sorted_thetas = SortedThetas(thetas={0: [np.array([1, 2.2]), np.array([1.2, 2.6])],
                                             1: [np.array([3, 4])]},
                                     labels=np.array([[1.1, 2.25], [3, 4]]),
                                     histogram={0: 2, 1: 1})
        matA = np.array([[1, 2], [3, 5]])
        raw_data = np.array([[1, 2.2], [1.2, 2.6]])
        data_matrix = raw_data.mean(axis=0) - raw_data
        cov_mat = self.analyzer.get_covariance(0, sorted_thetas)
        AssertAlmostEqualMatrices(cov_mat, COV_MAT)
        AssertAlmostEqualMatrices(data_matrix.T.dot(data_matrix)/2, COV_MAT)

    def test_get_total_covariance(self):
        cov_mat = self.analyzer.get_total_covariance(THETAS)
        AssertAlmostEqualMatrices(cov_mat,
                                  np.array([[0.8088889, 0.6888889],
                                            [0.6888889, 0.5955556]]))

    def test_pca_svd_with_data_matrix(self):
        analyzer = ThetasAnalyzer(8)
        data_matrix = analyzer.to_data_matrix(theta8x100.thetas, subtract_avgs=True)
        cov_matrix = (data_matrix.T.dot(data_matrix))/data_matrix.shape[0]
        U, sing_vals, W = analyzer._pca_reduction_svd(data_matrix, cutoff_ratio=0)# no cutoff
        built_data_matrix = U @ sing_vals @ W
        AssertAlmostEqualMatrices(data_matrix, built_data_matrix)
        AssertAlmostEqualMatrices(data_matrix.mean(axis=0), data_matrix) #'subtract_avgs == True' works as it should
        squared_sing_vals = sing_vals.T @ sing_vals
        AssertAlmostEqualMatrices(data_matrix, built_data_matrix)
        built_cov_matrix = (W.T @ squared_sing_vals @ W) / data_matrix.shape[0]
        AssertAlmostEqualMatrices(built_cov_matrix, cov_matrix)
        new_data_matrix = data_matrix @ W.T
        new_cov_matrix = (new_data_matrix.T.dot(new_data_matrix)) / new_data_matrix.shape[0]

        AssertAlmostEqualMatrices(new_cov_matrix, squared_sing_vals[0:8, :]/100)

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
        AssertAlmostEqualMatrices(U @ sing_mat @ W, data_matrix)
        AssertAlmostEqualMatrices(U.T @ U, np.eye(5))
        AssertAlmostEqualMatrices(W @ W.T, np.eye(4))
        AssertAlmostEqualMatrices(sing_mat, np.array([[2.7751688, 0, 0, 0, 0],
                                                      [0, 2.1213203, 0, 0, 0],
                                                      [0, 0, 1.13949018, 0, 0],
                                                      [0, 0, 0, 0, 0]]))
        cov_sings = sing_mat.T @ sing_mat
        reconstructed_cov = (W.T @ cov_sings @ W)/data_matrix.shape[0]
        AssertAlmostEqualMatrices((data_matrix.T @ data_matrix)/data_matrix.shape[0], reconstructed_cov)

    def test_pca_reduction_svd_with_8dim(self):
        analyzer = ThetasAnalyzer(8)
        cov_matrix = analyzer.get_total_covariance(theta8x100.thetas)
        data_matrix = analyzer.to_data_matrix(theta8x100.thetas, subtract_avgs=True)
        U, sing_vals, W = analyzer._pca_reduction_svd(data_matrix, cutoff_ratio=0.23)
        print(U.shape, sing_vals.shape, W.T.shape)
        reduced_cov = W @ sing_vals.T @ sing_vals @ W.T / data_matrix.shape[0]
        reduced_data_matrix = data_matrix @ W.T
        self.assertEqual(sing_vals.shape, (100, 3))
        AssertAlmostEqualMatrices(np.diagflat([5.57996498, 4.80015414, 0.32748969]), sing_vals)
        self.assertEqual(U.shape, (100, 100))
        self.assertEqual(W.shape, (8, 3))
        self.assertEqual(reduced_cov.shape, (8, 8))
        self.assertAlmostEqual(7.665660908, linalg.norm(cov_matrix-reduced_cov))
        AssertAlmostEqualMatrices(np.identity(3), U.T.dot(U))
        AssertAlmostEqualMatrices(np.identity(3), W.T.dot(W))
        print(np.abs(cov_matrix - reduced_cov).max())
        self.assertTrue(np.abs(cov_matrix - reduced_cov).max() < 1.9)
        print(cov_matrix)
        print(reduced_data_matrix.shape)
        reduced_cov = reduced_data_matrix.T.dot(reduced_data_matrix)/100
        print(reduced_cov)

        built_data_matrix = U @ sing_vals
        print(built_data_matrix - reduced_data_matrix)

    def test_get_cov_pca_reductions(self):
        analyzer = ThetasAnalyzer(8)
        thetas = {0: theta8x100.thetas, 1: theta8x100.thetas[50:], 2: []}
        sorted_thetas = SortedThetas(thetas=thetas, labels=[], histogram={})

        reduced_covs = analyzer.cov_pca_reductions(sorted_thetas, cutoff_ratio=0.05)
        cov_mat = analyzer.get_total_covariance(theta8x100.thetas)
        self.assertEqual(list(reduced_covs.keys()), [0, 1])
        U, sing_vals = reduced_covs[0]
        reduced_cov = U @ sing_vals @ U.T
        self.assertEqual(sing_vals.shape, (3, 3))
        AssertAlmostEqualMatrices(np.diagflat([5.57996498, 4.80015414, 0.32748969]), sing_vals)
        self.assertEqual(U.shape, (8, 3))
        self.assertEqual(reduced_cov.shape, (8, 8))
        self.assertAlmostEqual(0.26056294, linalg.norm(cov_mat - reduced_cov))
        AssertAlmostEqualMatrices(np.identity(3), U.T.dot(U))
        self.assertTrue(np.abs(cov_mat - reduced_cov).max() < .12)

    '''def test_solution_spaces(self):
        analyzer = ThetasAnalyzer(8)
        thetas = {0: theta8x100.thetas[0:10], 1: theta8x100.thetas[10:20], 2: []}
        first_label = sum(theta8x100.thetas[0:10])/10
        second_label = sum(theta8x100.thetas[10:20])/10
        sorted_thetas = SortedThetas(thetas=thetas, labels=[first_label, second_label], histogram={})


        sol_spaces = analyzer.solution_spaces(sorted_thetas, cutoff_ratio=0.005)
        cov_mat = analyzer.get_total_covariance(theta8x100.thetas)
        reduced_covs = analyzer.cov_pca_reductions(sorted_thetas, cutoff_ratio=0.000000000005)

        U0, red_sings = reduced_covs[0]
        eig00 = U0[:, 0]
        eig01 = U0[:, 1]
        reduced_cov = U0 @ red_sings @ U0.T
        self.assertEqual(list(sol_spaces.keys()), [0, 1])
        theta_mean, P = sol_spaces[0]['label'], sol_spaces[0]['projection']

        AssertAlmostEqualMatrices(P, P.T)
        AssertAlmostEqualMatrices(P, P.dot(P))

        data_matrix = analyzer.to_data_matrix(theta8x100.thetas[0:10], subtract_avgs=True)

        #print(data_matrix)
        #print(data_matrix-(P @ data_matrix.T).T)
        AssertAlmostEqualMatrices(analyzer.get_covariance(0, sorted_thetas), np.cov(data_matrix.T))
        new_data_matrix = data_matrix.dot(U0)
        print(data_matrix.shape)
        print(U0.shape)
        print(red_sings.shape)
        reduced_data = U0 @ np.sqrt(red_sings)
        self.assertEqual(reduced_data.shape, (10, 5))
        print(new_data_matrix.shape)
        self.assertEqual(new_data_matrix.shape, (data_matrix.shape[0], U0.shape[1])) #(10, 5) == shape

        self.assertEqual(reduced_data.shape, (5, 8))
        AssertAlmostEqualMatrices(np.cov(reduced_data), reduced_data.T @ reduced_data)
        print(reduced_cov - reduced_data @ reduced_data.T)
        cov_differences = np.cov(reduced_data) - np.cov(data_matrix.T)
        print(np.abs(cov_differences).max())
        print(np.abs(cov_differences).mean())
        #print(np.linalg.eig(P)[1][:, 1])'''


    def test_finding_svd(self):#OK
        analyzer = ThetasAnalyzer(8)
        data_matrix = analyzer.to_data_matrix(theta8x100.thetas, subtract_avgs=True)
        rows, cols = data_matrix.shape
        U, singular_values, W = linalg.svd(data_matrix)
        #recomputed_data_matrix = np.dot(U[:, :cols] * singular_values, W) #The shortest and quickest way
        diags = np.diagflat(singular_values)
        sing_mat = np.zeros(data_matrix.shape)
        sing_mat[0:diags.shape[0], 0:diags.shape[1]] = diags
        recomputed_data_matrix = np.dot(U, sing_mat.dot(W))
        AssertAlmostEqualMatrices(recomputed_data_matrix, data_matrix)

    def test_pca_reduction_svd(self):
        analyzer = ThetasAnalyzer(8)
        data_matrix = analyzer.to_data_matrix(theta8x100.thetas, subtract_avgs=True)
        cov_matrix = np.cov(data_matrix.T)
        U, sing_mat, W = analyzer.get_svd(cov_matrix)
        reconstructed_cov_mat = U @ sing_mat @ W
        should_be_zero = reconstructed_cov_mat - cov_matrix
        AssertAlmostEqualMatrices(reconstructed_cov_mat, cov_matrix)

    def test_distance_of_pca_decomp(self): #They should be zeros if cov_mat is of full rank
        analyzer = ThetasAnalyzer(8)
        data_matrix = analyzer.to_data_matrix(theta8x100.thetas, subtract_avgs=True)
        cov_matrix = np.cov(data_matrix.T)
        u, sing_mat, w = analyzer.get_svd(cov_matrix)
        eigen_values, eigen_vectors = linalg.eig(cov_matrix)
        sorted_eigen_values = np.sort(eigen_values, axis=0)
        eig_val_mat = np.diagflat(sorted_eigen_values)
        print(sorted_eigen_values, eigen_values)

        #AssertAlmostEqualMatrices(eig_val_mat, sing_mat)

    def tearDown(self):
        del self.analyzer

#-----------Help fns for testing how to create a fitting theta-generating function.---

from math import log, sin, asin, tan, sinh, asinh, sqrt, pow, atan

def center(x):
    '''Centers thetas in the interval [0,pi] symmetrically
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


class TestGDFTBuilder(unittest.TestCase):

    def setUp(self):
        self.builder = GDFTBuilder(8)

    def test_build(self):
        gdft = self.builder.build()
        should_be_identity = gdft.dot(np.conjugate(gdft.T))
        AssertAlmostEqualMatrices(should_be_identity, 8*np.eye(8))
        correlations = ThetasAnalyzer(8).get_correlations(gdft)
        print(correlations)


    def tearDown(self):
        del self.builder


class TestApproximatingMatrix(unittest.TestCase):

    def setUp(self):
        self.matrix = np.array([[0.09, -0.01], [0.99, 1.15]])
        self.complex_matrix = np.array([[0.02+1j*0.02, -0.01], [0.99-1j*0.001, 1.15+1j*0.04]], dtype=np.complex128)

    def test_small_els_to(self):
        result = small_els_to(self.matrix, replace_val=0, cutoff=0.10)
        self.assertTrue(EqualMatrices(result, np.array([[0, 0], [0.99, 1.15]])))
        AssertAlmostEqualMatrices(np.array([[0, 0], [0.113, 1.15]]), np.array([[0, 0], [0.11, 1.15]]), decimals=2)

    def test_approximate_matrix(self):
        result = approximate_matrix(self.matrix, tol=0.02)
        AssertAlmostEqualMatrices(result, np.array([[0.09, 0], [1, 1.15]]), decimals=2)

        AssertAlmostEqualMatrices(approximate_matrix(self.complex_matrix, tol=0.1),
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

class TestSymmetry(unittest.TestCase):

    def setUp(self):
        self.analyzer = SymmetryAnalyzer(8)
        self.gdft = gdft_matrix(8, THETAS8)

    def test_dependency_on_G2(self):
        gammas = np.pi*np.random.rand(8, 1)
        print(gammas)
        new_gdft = two_param_gdft_matrix(8, THETAS8, gammas)  # Correlations are not dependent on gammas!!

        self.assertEqual(self.analyzer.get_similarities(self.gdft, new_gdft, 0.01),
                         [True, True, True, True, True])
        should_be_8_matrix = new_gdft.dot(new_gdft.conjugate().transpose())

        AssertAlmostEqualMatrices(should_be_8_matrix, 8*np.identity(8))

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
