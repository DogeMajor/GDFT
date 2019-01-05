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

#------Test data-----------------------------------------------------------------------------
thetas16x30 = extract_thetas_records("data/", "30thetas_16x16__1-1_21_14.json")

normalized_thetas = np.array([-2.98774983e-09, 8.18550897e-01, 2.79042360e+00, 2.67879537e+00,
                              1.78476702e+00, 1.08366030e-01, 2.63164508e+00, 2.50189183e-02])

poly1 = [-7.48008226e-03,  1.73918516e-01, -1.61022589e+00,  7.60466637e+00,
         -1.93129846e+01,  2.45161101e+01, -1.05913241e+01,  3.61686052e-01]

poly2 = [-7.47996332e-03,  1.92602532e-01, -2.00262320e+00,  1.07213698e+01,
         -3.09003179e+01,  4.47562012e+01, -2.53956497e+01,  3.03837966e+00]

thetas_16gdft = np.array([0.47918196, 3.14159265, 0.37415556, 2.32611506, 0.77481029, 3.08069088,
                          2.36308541, 0.66752458, 2.7953271, 3.07615731, 0.29459556, 0.30038568,
                          0.,        0.,        3.14159265, 3.14159265])

orderings_example = np.array([9, 10, 13, 4, 12, 2, 0, 7, 6, 1, 11, 5, 8, 3, 14, 15])


class TestThetasAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = ThetasAnalyzer(16)

    def test_arg_orderings(self):
        orderings_mat = np.zeros((30, 16))
        for row, theta in enumerate(thetas16x30.thetas):
            orderings_mat[row, :] = theta.argsort()

        print(orderings_mat[np.logical_and(orderings_mat[:, 15] == 15, orderings_mat[:, 14] == 14)])
        print(orderings_mat.mean(axis=0))

    def test_finiding_seqs_in_thetas(self):
        finder = SequenceFinder()
        for theta in thetas16x30.thetas:
            theta = 16*(theta-np.pi/2)/np.pi
            print(2*np.array(finder.nth_diff(theta, 0)))

    def tearDown(self):
        del self.analyzer

#-----------Help fns for testing how to create a fitting theta-generating function.---

from math import log, sin, asin, tan, sinh, asinh, sqrt, pow, atan

def center(x):
    '''Centers thetas in the interval [0,pi] symmetrically
    with respect to point pi/2'''
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
        should_be_identity = gdft.dot(np.conjugate(gdft))
        self.assertTrue(AlmostEqualMatrices(should_be_identity, 8*np.eye(8)))
        correlations = ThetasAnalyzer(8).get_correlations(gdft)
        print(correlations)

    def test_build_by_calclulating_roots_by_hand(self):
        generator = RootGenerator(8)
        roots = [0, 7, 7-2*np.pi]
        complex_root_feeds = [7-2*np.pi + (3/5)*np.pi, 7-2*np.pi + (4/5)*np.pi]
        for x in complex_root_feeds:
            roots.append(generator.polynome_root(x))
            roots.append(generator.polynome_root(x).conjugate())
        poly = np.poly1d(roots, True)
        thetas = [poly(n) for n in range(8)]
        print("Thetas", thetas)
        gdft = gdft_matrix(8, np.array(thetas))
        should_be_identity = gdft.dot(np.conjugate(gdft))
        self.assertTrue(AlmostEqualMatrices(should_be_identity, 8*np.eye(8)))
        correlations = ThetasAnalyzer(8).get_correlations(gdft)
        print(correlations)

    def tearDown(self):
        del self.builder


if __name__ == '__main__':
    #pass
    unittest.main()