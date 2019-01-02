import sys
sys.path.append("../src")
sys.path.append("src/")
import unittest
import numpy as np
from tools import *
from gdft import *
from correlations import *
from analyzer import *
from sequencefinder import SequenceFinder

#------Test data-----------------------------------------------------------------------------

normalized_thetas = np.array([-2.98774983e-09, 8.18550897e-01, 2.79042360e+00, 2.67879537e+00,
                              1.78476702e+00, 1.08366030e-01, 2.63164508e+00, 2.50189183e-02])

poly1 = [-7.48008226e-03,  1.73918516e-01, -1.61022589e+00,  7.60466637e+00,
         -1.93129846e+01,  2.45161101e+01, -1.05913241e+01,  3.61686052e-01]

poly2 = [-7.47996332e-03,  1.92602532e-01, -2.00262320e+00,  1.07213698e+01,
         -3.09003179e+01,  4.47562012e+01, -2.53956497e+01,  3.03837966e+00]

thetas_16gdft = np.array([0.47918196, 3.14159265, 0.37415556, 2.32611506, 0.77481029, 3.08069088,
                          2.36308541, 0.66752458, 2.7953271, 3.07615731, 0.29459556, 0.30038568,
                          0.,        0.,        3.14159265, 3.14159265])

class TestThetasAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = ThetasAnalyzer(8)

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
    return [0.5 * np.pi + 1.135 * atan(n - 2*3.63) for n in range(16)]

class TestFindingThetaGeneratingFunction(unittest.TestCase):
    '''Mostly just visual tests to experiment how to build the GDFT from
    the function approximation with permutation matrices.
    Not unit/integration/system tests by any measure!'''

    def setUp(self):
        self.seq_finder = SequenceFinder()
        self.centered_thetas = transform(normalized_thetas[0:], center)#lulzs

    def test_generating_thetas_with_arctan(self):
        generated_thetas = generate_thetas()
        norm = seq_norm(generated_thetas, sorted(self.centered_thetas))
        print("l^2-norm for the ordered thetas", norm)
        self.assertTrue(norm < 0.3)

    def test_ordering_generated_thetas_with_permutation(self):
        perm = permutation_matrix(8, orderings=[0, 3, 7, 6, 4, 2, 5, 1])
        permutated_thetas = perm.dot(generate_thetas())
        print("permutated_ thetas", permutated_thetas)
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
        self.assertTrue(norm < 1.0)

    def tearDown(self):
        del self.seq_finder
        del self.centered_thetas

'''gdft = gdft_matrix(16, thetas_16gdft)
print(gdft.shape)
correlations = ThetasAnalyzer(16).get_correlations(gdft)
print(correlations.avg_auto_corr)
print(correlations)'''


class TestRootGenerator(unittest.TestCase):

    def setUp(self):
        self.generator = RootGenerator(8)

    def test_ellipsis_height(self):
        self.assertEqual(self.generator.ellipsis_height(2), 0)
        self.assertEqual(self.generator.ellipsis_height(3.5), 1)
        self.assertEqual(self.generator.ellipsis_height(2.5), np.sqrt(1 - 1/1.5**2))

    def test_polynome_roots(self):
        poly_roots = self.generator.polynome_roots()
        print(poly_roots)


    def tearDown(self):
        del self.generator


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

def to_real(seq):
    return [item.real for item in seq]

def to_imag(seq):
    return [item.imag for item in seq]

def filter_zeros(seq):
    return [item for item in seq if item != 0]

def to_slopes(seq):
    reals = filter_zeros(to_real(seq))
    imags = filter_zeros(to_imag(seq))


if __name__ == '__main__':
    #pass
    unittest.main()