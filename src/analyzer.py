import time
from collections import namedtuple, Counter
from math import atan, isclose
import numpy as np
from numpy import linalg
from scipy.cluster.vq import kmeans2
from gdft import gdft_matrix
from correlations import CorrelationAnalyzer

np.random.seed(int(time.time()))

Polynomes = namedtuple('Polynomes', 'polynomes theta_vecs')
SortedThetas = namedtuple('SortedThetas', 'thetas labels histogram correlations')
SortedPolynomes = namedtuple('SortedPolynomes', 'polynomes kmean_labels')


class Classifier(object):


    def get_correlations(self, grouped_theta_vecs):
        dim = grouped_theta_vecs[0][0].shape[0]
        corr_analyzer = CorrelationAnalyzer(dim)
        def get_corrs(thetas):
            corrs = []
            for theta in thetas:
                gdft_mat = gdft_matrix(dim, theta)
                corrs.append(corr_analyzer.get_correlations(gdft_mat))
            return corrs

        return {label_ind: get_corrs(theta_vecs) for label_ind, theta_vecs in grouped_theta_vecs.items()}

    def sort_thetas(self, theta_vecs, groups):
        old_kmeans_results = self._classify_thetas(theta_vecs, groups)

        new_kmeans_results = self.filter_empty_labels(old_kmeans_results)
        grouped_theta_vecs = self.group_by_label(theta_vecs, new_kmeans_results)
        grouped_corrs = self.get_correlations(grouped_theta_vecs)
        #print("grouped_corrs:", grouped_corrs)
        return SortedThetas(thetas=grouped_theta_vecs, labels=new_kmeans_results[0],
                            histogram=self._kmeans_to_histogram(new_kmeans_results), correlations=grouped_corrs)

    def filter_empty_labels(self, kmeans_results):
        labels, labelings = kmeans_results[0], kmeans_results[1]
        hist = self._kmeans_to_histogram(kmeans_results)
        non_empty_labels = sorted([key for key in hist.keys() if hist[key] != []])
        labeling_transforms = {old_labeling: new_labeling for new_labeling, old_labeling in enumerate(non_empty_labels)}
        new_labels = labels[non_empty_labels, :]
        new_labelings = [labeling_transforms[old_labeling] for old_labeling in labelings]
        return new_labels, np.array(new_labelings)

    def _classify_thetas(self, theta_vecs, groups):
        return kmeans2(theta_vecs, groups)

    def group_by_label(self, unsorted_thetas, k_means_results):
        labels = k_means_results[1]
        sorted_thetas = {}
        for ind, theta in enumerate(unsorted_thetas):
            label = labels[ind]
            sorted_thetas.setdefault(label, []).append(theta)

        return sorted_thetas

    def _kmeans_to_histogram(self, k_means_results):
        return Counter(k_means_results[1])


class PCA(object):

    def __init__(self, dim):
        self._dim = dim

    def to_data_matrix(self, thetas, subtract_avgs=False):
        '''Changes a list of 1-D numpy arrays into a data matrix
        of shape data_points x measurement_dimension'''
        shape = (len(thetas), self._dim)
        records = np.concatenate(thetas).reshape(shape)
        if subtract_avgs:
            records = records - records.mean(axis=0)
        return records

    def get_svd(self, data_matrix):
        '''Returns U and W.T directly (NOT W!!) + singular vals
        as a list'''
        U, sing_values, W = linalg.svd(data_matrix)
        diags = np.diagflat(sing_values)
        return U, diags, W

    def get_total_covariance(self, thetas):
        length = len(thetas)
        thetas_matrix = self.to_data_matrix(thetas, subtract_avgs=True)
        return thetas_matrix.T.dot(thetas_matrix)/length

    def get_covariance(self, label_no, sorted_thetas):
        thetas = sorted_thetas.thetas[label_no]
        return self.get_total_covariance(thetas)

    def _pca_reduction_svd(self, mat, cutoff_ratio=0):
        U, sing_vals, W = linalg.svd(mat)
        max_sing = max(np.abs(sing_vals))
        mask = [index for index, eig in enumerate(sing_vals) if np.abs(eig)/max_sing > cutoff_ratio]
        sing_diag = np.diagflat(sing_vals[mask])
        return U[:, mask], sing_diag, W[mask, :]

    def cov_pca_reduction(self, label_no, sorted_thetas, cutoff_ratio=0):
        #cov_matrix = self.get_covariance(label_no, sorted_thetas)
        thetas = sorted_thetas.thetas[label_no]
        data_matrix = self.to_data_matrix(thetas, subtract_avgs=True)
        U, sing_vals, W = self._pca_reduction_svd(data_matrix, cutoff_ratio=cutoff_ratio)
        sing_mat = sing_vals.T @ sing_vals
        return W, sing_mat

    def cov_pca_reductions(self, sorted_thetas, cutoff_ratio=0):
        def is_empty(val):
            return val == []
        non_empty_labels = (key for key, val in sorted_thetas.thetas.items() if not is_empty(val))
        return {key: self.cov_pca_reduction(key, sorted_thetas, cutoff_ratio=cutoff_ratio)
                for key in non_empty_labels}


class ThetasAnalyzer(object):

    def __init__(self, dim):
        self._dim = dim
        self._corr_analyzer = CorrelationAnalyzer(dim)

    def get_correlations(self, gdft):
        return self._corr_analyzer.get_correlations(gdft)

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

    def sort_thetas(self, theta_vecs, groups):
        classifier = Classifier()
        sorted_thetas = classifier.sort_thetas(theta_vecs, groups)
        return sorted_thetas

    def cov_pca_reductions(self, sorted_thetas, cutoff_ratio=0):
        pca = PCA(self._dim)
        return pca.cov_pca_reductions(sorted_thetas, cutoff_ratio=cutoff_ratio)

    def solution_spaces(self, sorted_thetas, cutoff_ratio=0):
        '''We've chosen the right 'eigen space' and therefore
         projections operate from right side on (row) vectors  (==data)'''
        pca = PCA(self._dim)
        pca_reductions = pca.cov_pca_reductions(sorted_thetas, cutoff_ratio=cutoff_ratio)

        def get_solution_space(key):
            W = pca_reductions[key][0]
            return {'label': sorted_thetas.labels[key], 'projection': W.T @ W}

        sol_spaces = {key: get_solution_space(key) for key in pca_reductions.keys()}
        return sol_spaces


class SymmetryAnalyzer(object):

    def __init__(self, dim):
        self._dim = dim
        self._corr_analyzer = CorrelationAnalyzer(self._dim)

    def get_correlations(self, gdft):
        return self._corr_analyzer.get_correlations(gdft)

    def get_similarities(self, old_gdft, new_gdft, rel_tol=10e-9):
        old_correlations = self.get_correlations(old_gdft)
        new_correlations = self.get_correlations(new_gdft)
        similarities = (isclose(old_corr, new_corr, rel_tol=rel_tol) for old_corr, new_corr
                        in zip(old_correlations, new_correlations))
        return list(similarities)

    def get_symmetry(self, old_gdft, new_gdft, rel_tol=10e-9):
        similarities = self.get_similarities(old_gdft, new_gdft, rel_tol=rel_tol)
        return sum(similarities) == len(similarities)

if __name__ == "__main__":
    pass