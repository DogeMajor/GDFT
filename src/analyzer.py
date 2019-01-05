import time
from collections import namedtuple, Counter
import numpy as np

from scipy.cluster.vq import kmeans2
from utils import extract_thetas_records
from gdft import *
from correlations import *

np.random.seed(int(time.time()))


'''def extract_thetas_records(path, file_name):
    dao = DAO(path)
    content = dao.read(file_name)
    theta_vecs = [np.array(result["theta_vec"]) for result in content["results"]]
    corrs = [result["correlation"] for result in content["results"]]
    return Thetas(thetas=theta_vecs, correlations=corrs)'''

Polynomes = namedtuple('Polynomes', 'polynomes theta_vecs')
SortedThetas = namedtuple('SortedThetas', 'thetas labels histogram')
SortedPolynomes = namedtuple('SortedPolynomes', 'polynomes kmean_labels')


class ThetasAnalyzer(object):

    def __init__(self, dim):
        self._dim = dim
        self._analyzer = CorrelationAnalyzer(dim)
        self._corr_fns = {"max_auto_corr": self._analyzer.max_auto_corr,
                          "avg_auto_corr": self._analyzer.avg_auto_corr,
                          "max_cross_corr": self._analyzer.max_cross_corr,
                          "avg_cross_corr": self._analyzer.avg_cross_corr,
                          "avg_merit_factor": self._analyzer.avg_merit_factor}

    def get_correlations(self, gdft):
        corr_obj = Correlation(gdft)
        self._analyzer.set_corr_tensor(corr_obj.correlation_tensor())
        Correlations = namedtuple('Correlations', self._corr_fns.keys())
        corrs = {fn_name: corr_fn() for fn_name, corr_fn in self._corr_fns.items()}
        return Correlations(**corrs)

    def get_theta_vecs(self, path, file_name):
        return extract_thetas_records(path, file_name)

    def _generate_points(self, theta_vec):
        length = theta_vec.shape[0]
        args = np.array(list(range(length)))
        return args, theta_vec

    def _fit_polynome(self, theta_vec, grade):
        args, thetas = self._generate_points(theta_vec)
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


class GDFTBuilder(object):

    def __init__(self, dim):
        self._dim = dim

    def _get_polynome(self):
        generator = RootGenerator(self._dim)
        roots = generator.polynome_roots()
        return np.poly1d(roots, True)

    def build(self):
        polynome = self._get_polynome()
        print(polynome)
        shifts = np.array([polynome(n) for n in range(self._dim)])
        return gdft_matrix(self._dim, shifts)

