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

normalized_thetas = np.array([-2.98774983e-09, 8.18550897e-01, 2.79042360e+00, 2.67879537e+00,
                              1.78476702e+00, 1.08366030e-01, 2.63164508e+00, 2.50189183e-02])

poly1 = [-7.48008226e-03,  1.73918516e-01, -1.61022589e+00,  7.60466637e+00,
         -1.93129846e+01,  2.45161101e+01, -1.05913241e+01,  3.61686052e-01]
fitted_poly = np.poly1d(poly1)

poly2 = [-7.47996332e-03,  1.92602532e-01, -2.00262320e+00,  1.07213698e+01,
         -3.09003179e+01,  4.47562012e+01, -2.53956497e+01,  3.03837966e+00]
fitted_poly2 = np.poly1d(poly2)


fitted_poly3 = np.poly1d([7.47998801e-03, -1.92603126e-01,  2.00262887e+00, -1.07213972e+01,
                          3.09003879e+01, -4.47562920e+01,  2.54258825e+01,  7.87788234e-02])

fitted_poly4 = np.poly1d([-7.47993194e-03,  1.73914814e-01, -1.61019005e+00,  7.60449390e+00,
                          -1.93125564e+01,  2.45155999e+01, -1.05452314e+01,  2.51692087e-01])

fitted_poly5 = np.poly1d([-7.47996777e-03,  1.73915780e-01, -1.61020024e+00,  7.60454701e+00,
                          -1.93126975e+01,  2.45157763e+01, -1.05627678e+01,  3.86084940e-01])

fitted_poly6 = np.poly1d([-7.47998681e-03,  1.73916249e-01, -1.61020475e+00,  7.60456850e+00,
                          -1.93127506e+01,  2.45158373e+01, -1.04651218e+01,  2.87400700e-02])


thetas_16gdft = np.array([0.47918196, 3.14159265, 0.37415556, 2.32611506, 0.77481029, 3.08069088,
                          2.36308541, 0.66752458, 2.7953271, 3.07615731, 0.29459556, 0.30038568,
                          0.,        0.,        3.14159265, 3.14159265])

finder = SequenceFinder()


from math import log, sin, asin, tan, sinh, asinh, sqrt, pow, atan

def center(x):
    return x + 0.175584

def func(x):
    return 0.5*np.pi + atan(center(x))

def transform(seq, fn):
    return [fn(item) for item in seq]

def seq_norm(seq_a, seq_b):
    distances = ((item_b - item_a)**2 for item_a, item_b in zip(seq_a, seq_b))
    return sqrt(sum(distances))

transformed_seq = transform(normalized_thetas[0:], center)

print(sorted(transformed_seq))

'''for n in range(8):
    print(finder.nth_diff(transformed_seq, n))
    print(finder.nth_diff(sorted(transformed_seq), n))'''

generated_thetas = [0.5*np.pi + 1.135*atan(n-3.63) for n in range(8)]
print("l^2-norm for the ordered thetas", seq_norm(generated_thetas, sorted(transformed_seq)))

#perm = permutation_matrix(8, orderings=[0, 2, 7, 6, 4, 3, 5, 1])
perm = permutation_matrix(8, orderings=[0, 3, 7, 6, 4, 2, 5, 1])
permutated_thetas = perm.dot(generated_thetas)

print("l^2 norm metric", seq_norm(transformed_seq, permutated_thetas))

gdft = gdft_matrix(8, permutated_thetas)
correlations = ThetasAnalyzer(8).get_correlations(gdft)
print(correlations.avg_auto_corr)
print(transformed_seq)
print(permutated_thetas)


gdft = gdft_matrix(16, thetas_16gdft)
print(gdft.shape)
correlations = ThetasAnalyzer(16).get_correlations(gdft)
print(correlations.avg_auto_corr)
print(correlations)


class TestThetasAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = ThetasAnalyzer(8)

    def tearDown(self):
        del self.analyzer


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
    pass
    #unittest.main()