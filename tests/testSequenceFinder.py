import sys
sys.path.append("../src")
sys.path.append("src/")
import unittest
import numpy as np
from tools import *
from gdft import *
from correlations import *
from sequencefinder import *

GDFT_MAT = np.array([[1, -1], [-1, -1]], dtype=np.complex128)

class TestSequenceFinder(unittest.TestCase):

    def setUp(self):
        self.corrs = Correlation(dft_matrix(2)).correlation_tensor()
        self.finder = SequenceFinder()

    def test_to_integers(self):
        seq = np.array([-0.1, 0.2, -0.3, 0.4, -0.5])
        integers = self.finder._to_integers(seq)
        self.assertEqual(integers, [-1, 2, -3, 4, -5])

    def test_nth_diff(self):
        seq = [-1, 2, -3, 4, -5]
        self.assertEqual(self.finder.nth_diff(seq, 0), seq)
        self.assertEqual(self.finder.nth_diff([0], 5), [])
        first_diffs = self.finder.nth_diff(seq, 1)
        self.assertEqual(first_diffs, [3, -5, 7, -9])
        second_diffs = self.finder.nth_diff(seq, 2)
        self.assertEqual(second_diffs, [-8, 12, -16])
        third_diffs = self.finder.nth_diff(seq, 3)
        self.assertEqual(third_diffs, [20, -28])
        fourth_diffs = self.finder.nth_diff(seq, 4)
        self.assertEqual(fourth_diffs, [-48])
        fifth_diffs = self.finder.nth_diff(seq, 5)
        self.assertEqual(fifth_diffs, [])

    def tearDown(self):
        del self.corrs
        del self.finder


if __name__ == '__main__':
    unittest.main()
