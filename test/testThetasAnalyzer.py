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

COV_MAT = np.array([[0.02, 0.04], [0.04, 0.08]])

THETAS = [np.array([1, 2.2]), np.array([1.2, 2.6]), np.array([3, 4])]

class TestThetasAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = ThetasAnalyzer(2)

    def test_arg_orderings(self):
        orderings_mat = np.zeros((30, 16))
        for row, theta in enumerate(thetas16x30.thetas):
            orderings_mat[row, :] = theta.argsort()

        #print(orderings_mat[np.logical_and(orderings_mat[:, 15] == 15, orderings_mat[:, 14] == 14)])
        #print(orderings_mat.mean(axis=0))

    def test_finding_seqs_in_thetas(self):
        finder = SequenceFinder()
        for theta in thetas16x30.thetas:
            theta = 16*(theta-np.pi/2)/np.pi
            #print(2*np.array(finder.nth_diff(theta, 0)))

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
        cov_mat = self.analyzer.get_covariance(0, sorted_thetas)
        AssertAlmostEqualMatrices(cov_mat, COV_MAT)

    def test_get_total_covariance(self):
        cov_mat = self.analyzer.get_total_covariance(THETAS)
        AssertAlmostEqualMatrices(cov_mat, np.array([[1.21333333, 1.03333333], [1.03333333, 0.89333333]]))

    def test_pca_reduction_eig(self):
        eig_values, eig_vectors = self.analyzer._pca_reduction_eig(COV_MAT, cutoff_ratio=0.05)
        self.assertEqual(eig_values, np.array([[0.1]]))
        AssertAlmostEqualMatrices(eig_vectors, np.array([[-0.4472136], [-0.89442719]]))

    def test_construct_diagonalized_cov_mat(self):#OK
        analyzer = ThetasAnalyzer(8)
        diags, w = analyzer.get_cov_svd(theta8x100.thetas)
        data_matrix = analyzer.to_data_matrix(theta8x100.thetas, subtract_avgs=True)
        U, sing_mat, W = analyzer.get_svd(data_matrix)
        new_data_matrix = np.dot(data_matrix, W)
        new_cov_mat = np.cov(new_data_matrix)
        AssertAlmostEqualMatrices(np.dot(sing_mat.T, sing_mat), new_cov_mat)
        should_be_zero = new_cov_mat
        np.fill_diagonal(should_be_zero, 0)
        AssertAlmostEqualMatrices(np.zeros((8, 8)), should_be_zero)

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

    def test_get_cov_svd(self):
        analyzer = ThetasAnalyzer(8)
        data_matrix = analyzer.to_data_matrix(theta8x100.thetas, subtract_avgs=True)
        cov_matrix = np.cov(data_matrix.T)
        u, sing_mat, w = analyzer.get_svd(cov_matrix)
        diagonalized_cov, W = analyzer.get_cov_svd(theta8x100.thetas)
        # AssertAlmostEqualMatrices(diagonalized_cov, sing_mat)
        print((W.T).shape)
        print(diagonalized_cov.shape)
        print(W.shape)
        reconstructed_cov_mat = (W).dot(diagonalized_cov.dot(W.T))
        should_be_zero = reconstructed_cov_mat - cov_matrix
        # print(should_be_zero)
        AssertAlmostEqualMatrices(reconstructed_cov_mat, cov_matrix)

    def test_if_cov_svd_and_eig_are_same(self): #They should be if cov_mat is of full rank
        analyzer = ThetasAnalyzer(8)
        data_matrix = analyzer.to_data_matrix(theta8x100.thetas, subtract_avgs=True)
        cov_matrix = np.cov(data_matrix.T)
        u, sing_mat, w = analyzer.get_svd(cov_matrix)
        eigen_values, eigen_vectors = linalg.eig(cov_matrix)
        orderings = np.argsort(eigen_values)
        print(orderings)
        perm = permutation_matrix(8, orderings=orderings[::-1])
        #print(perm)
        print(eigen_values.dot(perm))
        sorted_eigen_values = np.sort(eigen_values, axis=0)
        eig_val_mat = np.diagflat(sorted_eigen_values[::-1])
        AssertAlmostEqualMatrices(eig_val_mat, sing_mat)

    def test_distance_of_pca_decomp(self): #They should be if cov_mat is of full rank
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
        gammas = np.random.rand(8, 1)
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
