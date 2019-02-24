import time
from collections import namedtuple, Counter
from math import atan, isclose
import numpy as np
from scipy.cluster.vq import kmeans2
from utils import extract_thetas_records
from gdft import gdft_matrix
from correlations import Correlation, CorrelationAnalyzer

np.random.seed(int(time.time()))

Polynomes = namedtuple('Polynomes', 'polynomes theta_vecs')
SortedThetas = namedtuple('SortedThetas', 'thetas labels histogram')
SortedPolynomes = namedtuple('SortedPolynomes', 'polynomes kmean_labels')

class ThetasAnalyzer(object):

    def __init__(self, dim):
        self._dim = dim
        self._corr_analyzer = CorrelationAnalyzer(dim)

    def get_correlations(self, gdft):
        return self._corr_analyzer.get_correlations(gdft)

    def get_theta_vecs(self, path, file_name):
        return extract_thetas_records(path, file_name)

    def _generate_points(self, theta_vec):
        length = theta_vec.shape[0]
        args = np.array(list(range(length)))
        return args, theta_vec

    def _fit_polynome(self, theta_vec, grade):
        args, _ = self._generate_points(theta_vec)
        z = np.polyfit(args, theta_vec, grade)
        return np.poly1d(z)

    def fit_polynomes(self, thetas, grade):
        polynomes = [self._fit_polynome(theta_vec, grade) for theta_vec in thetas]
        return Polynomes(polynomes=polynomes, theta_vecs=thetas)

    def _fit_polynome_to(self, all_thetas, grade):
        thetas = np.core.records.fromarrays(all_thetas)
        length = thetas.shape[1]
        print(length)
        args = np.array(list(range(length)))

    def fit_sorted_polynomes(self, sorted_thetas):
        pass

    def sort_thetas(self, theta_vecs, groups):
        kmeans_results = self._classify_thetas(theta_vecs, groups)
        grouped_theta_vecs = self.group_by_label(theta_vecs, kmeans_results)
        return SortedThetas(thetas=grouped_theta_vecs, labels=kmeans_results[0],
                            histogram=self._to_histogram(kmeans_results))

    def _classify_thetas(self, theta_vecs, groups):
        return kmeans2(theta_vecs, groups)

    def group_by_label(self, unsorted_thetas, k_means_results):
        labels = k_means_results[1]
        sorted_thetas = {}
        for ind, theta in enumerate(unsorted_thetas):
            label = labels[ind]
            sorted_thetas.setdefault(label, []).append(theta)

        return sorted_thetas

    def _to_histogram(self, k_means_results):
        return Counter(k_means_results[1])


def generate_thetas_v1(dim):
    if dim <= 8:
        return [0.5 * np.pi + 1.135 * atan(n - 3.63) for n in range(dim)]
    return [0.5 * np.pi + 1.0 * atan(n - (dim/8) * 3.75) for n in range(dim)]

def generate_thetas(dim):
    poly = np.poly1d([-4.88506, 5.82857, -1.23057, 0.22106, -0.0184515])
    print(poly)
    return [poly(n) for n in range(dim)]

class GDFTBuilder(object):

    def __init__(self, dim):
        self._dim = dim

    def build(self):
        shifts = np.array(generate_thetas(self._dim))
        return gdft_matrix(self._dim, shifts)


class SymmetryAnalyzer(object):

    def __init__(self, dim):
        self._dim =dim
        self._corr_analyzer = CorrelationAnalyzer(self._dim)

    def get_correlations(self, gdft):
        return self._corr_analyzer.get_correlations(gdft)

    def get_similarities(self, old_gdft, new_gdft, rel_tol=10e-9):
        old_correlations = self.get_correlations(old_gdft)
        new_correlations = self.get_correlations(new_gdft)
        similarities = (isclose(old_corr, new_corr) for old_corr, new_corr in zip(old_correlations, new_correlations))
        return list(similarities)

    def get_symmetry(self, old_gdft, new_gdft, rel_tol=10e-9):
        similarities = self.get_similarities(old_gdft, new_gdft, rel_tol=rel_tol)
        return sum(similarities) == len(similarities)