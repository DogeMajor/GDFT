import sys
sys.path.append("../src")
sys.path.append("src/")
import unittest
import numpy as np
from utils import extract_thetas_records
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
