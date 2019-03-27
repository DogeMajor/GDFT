import unittest
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


def EqualMatrices(matA, matB):
    if matA.shape != matB.shape:
        return False
    return (matA == matB).all()


def AssertAlmostEqualMatrices(matA, matB, decimals=7):
    if matA.shape != matB.shape:
        return False
    return np.testing.assert_almost_equal(matA, matB, decimal=decimals)


def show_plot(wait=3, close=True):

    def _show_plot(fn):
        title = fn.__name__

        def plot_fn(self, *args, **kwargs):
            fn(self, *args, **kwargs)
            plt.title(title)
            plt.pause(wait)
            if close:
                plt.close()
            plt.show()
        return plot_fn
    return _show_plot

class GDFTTestCase(unittest.TestCase):

    def assertAlmostEqualLists(self, first_list, second_list, places=7):
        self.assertEqual(len(first_list), len(second_list))
        for first_item, second_item in zip(first_list, second_list):
            self.assertAlmostEqual(first_item, second_item, places=places)

    def assertEqualMatrices(self, matA, matB):
        if matA.shape != matB.shape:
            return False
        return np.testing.assert_equal(matA, matB)

    def assertAlmostEqualMatrices(self, matA, matB, decimals=7):
        if matA.shape != matB.shape:
            return False
        return np.testing.assert_almost_equal(matA, matB, decimal=decimals)

    def assertEqualCorrelationGroups(self, first_groups, second_groups):
        '''corr_group = {0: [Correlation(*), Corr...],...}'''
        #self.assertEqual(Counter(first_groups), Counter(second_groups))
        def same_corrs(first_corr_vecs, second_corr_vecs):
            for first_vec, second_vec in zip(first_corr_vecs, second_corr_vecs):
                self.assertTrue(first_vec == second_vec)

        for key in first_groups.keys():
            same_corrs(first_groups[key], second_groups[key])