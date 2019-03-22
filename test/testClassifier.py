import sys
import unittest
import numpy as np
sys.path.append("../src")
sys.path.append("src/")
from analyzer import Classifier
from tools import GDFTTestCase

#----------------------------Test data---------------------------------

KMEANS_RESULTS = (np.array([[-3.05, -4], [1, 2.05]]), np.array([1, 0]))

UNSORTED_THETAS = [np.array([1, 2.05]), np.array([-3.05, -4])]


class TestClassifier(GDFTTestCase):

    def setUp(self):
        self.classifier = Classifier()

    def test_sort_thetas(self):
        sorted_thetas = self.classifier.sort_thetas(UNSORTED_THETAS, 2)
        self.assertAlmostEqualMatrices(np.sort(sorted_thetas.labels, axis=0), KMEANS_RESULTS[0])
        self.assertEqual(sorted(list(sorted_thetas.histogram.values())), [1, 1])

    def test_group_by_label(self):
        groups = self.classifier.group_by_label(UNSORTED_THETAS, KMEANS_RESULTS)
        vals = list(groups.values())
        self.assertAlmostEqualMatrices(vals[0][0], np.array([1, 2.05]))
        self.assertAlmostEqualMatrices(vals[1][0], np.array([-3.05, -4]))
        self.assertTrue(len(vals[0]) == len(vals[0]) == 1)

    def test_kmeans_to_histogram(self):
        hist = self.classifier._kmeans_to_histogram(KMEANS_RESULTS)
        self.assertEqual(hist, {0: 1, 1: 1})

    def tearDown(self):
        del self.classifier

if __name__ == '__main__':
    unittest.main()
