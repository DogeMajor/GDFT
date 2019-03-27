import sys
import unittest
from random import shuffle
from math import sqrt, pow, atan
import numpy as np
sys.path.append("../src")
sys.path.append("src/")
from gdft import gdft_matrix, permutation_matrix, two_param_gdft_matrix
from correlations import *
from analyzer import ThetasAnalyzer, SymmetryAnalyzer
from sequencefinder import SequenceFinder
from tools import GDFTTestCase

#------Test data-----------------------------------------------------------------------------


NORMALIZED_THETAS = np.array([-2.98774983e-09, 8.18550897e-01, 2.79042360e+00, 2.67879537e+00,
                              1.78476702e+00, 1.08366030e-01, 2.63164508e+00, 2.50189183e-02])

THETAS_DIM_16 = np.array([0.47918196, 3.14159265, 0.37415556, 2.32611506, 0.77481029, 3.08069088,
                          2.36308541, 0.66752458, 2.7953271, 3.07615731, 0.29459556, 0.30038568,
                          0, 0, 3.14159265, 3.14159265])

#-----------Help fns for testing how to create a fitting theta-generating function.---

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
        self.centered_thetas = np.array(transform(NORMALIZED_THETAS, center))#lulzs

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
        norm = seq_norm(generated_thetas, sorted(THETAS_DIM_16))
        print("l^2-norm for the ordered thetas (len = 16)", norm)
        self.assertTrue(norm < 0.7)

    def test_ordering_generated_thetas_16_with_permutation(self):

        orderings = THETAS_DIM_16.argsort()
        print(orderings)
        self.assertEqual(list(orderings), [12, 13, 10, 11, 2, 0, 7, 4, 3, 6, 8, 9, 5, 1, 14, 15])
        perm = permutation_matrix(16, orderings=orderings)
        permutated_thetas = perm.dot(generate_thetas_16())
        norm = seq_norm(THETAS_DIM_16, permutated_thetas)
        print("l^2 norm metric for 16 len theta vector", norm)
        self.assertTrue(norm < 0.7)

    def tearDown(self):
        del self.seq_finder
        del self.centered_thetas


class TestSymmetryAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = SymmetryAnalyzer(16)
        self.gdft = gdft_matrix(16, THETAS_DIM_16)

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
THETAS8 = np.array([0.15404388, 2.74832147, 0.21274025, 1.87681229,
                    2.75850199, 2.85781138, 0.87359988, 0.04272007])

class TestSymmetry(GDFTTestCase):

    def setUp(self):
        self.analyzer = SymmetryAnalyzer(8)
        self.gdft = gdft_matrix(8, THETAS8)

    def test_dependency_on_G2(self):
        gammas = np.pi*np.random.rand(8, 1)
        print(gammas)
        new_gdft = two_param_gdft_matrix(8, gammas, THETAS8)
        # Correlations are not dependent on gammas!!

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
    unittest.main()
