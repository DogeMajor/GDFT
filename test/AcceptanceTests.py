import sys
sys.path.append("../src")
sys.path.append("src/")
import unittest
import datetime
from functools import wraps
from unittest import mock
from unittest.mock import patch
import numpy as np
from dao import DAO
from utils import extract_thetas_records, datetime_encoder
from tools import show_plot
from gdft import *
from correlations import *
from analyzer import *
from optimizer import Runner
from visualizer import *

#------Test data-----------------------------------------------------------------------------

'''thetas16x30 = extract_thetas_records("../data/", "30thetas_16x16__1-1_21_14.json")

normalized_thetas = np.array([-2.98774983e-09, 8.18550897e-01, 2.79042360e+00, 2.67879537e+00,
                              1.78476702e+00, 1.08366030e-01, 2.63164508e+00, 2.50189183e-02])

poly1 = [-7.48008226e-03,  1.73918516e-01, -1.61022589e+00,  7.60466637e+00,
         -1.93129846e+01,  2.45161101e+01, -1.05913241e+01,  3.61686052e-01]

thetas_16gdft = np.array([0.47918196, 3.14159265, 0.37415556, 2.32611506, 0.77481029, 3.08069088,
                          2.36308541, 0.66752458, 2.7953271, 3.07615731, 0.29459556, 0.30038568,
                          0.,        0.,        3.14159265, 3.14159265])

orderings_example = np.array([9, 10, 13, 4, 12, 2, 0, 7, 6, 1, 11, 5, 8, 3, 14, 15])
'''

#----mocked methods/fns/classes--------

def save_test_data(file_name, results):
    date_string = "today"
    dao = DAO("testdata/")
    dao.write(file_name + date_string + ".json", results)

class TestWithSmallSize(unittest.TestCase):
    '''Testing gdft builder with 4x4 matrices'''
    def setUp(self):
        self.runner = Runner(4)

    def test_generating_one_gdft(self):
        results = self.runner.optimize("avg_auto_corr", 1)
        records = results['results']
        self.assertTrue(records is not [])
        theta_vec = records[0]['theta_vec']
        correlation = records[0]['correlation']
        self.assertTrue(0.1*np.pi < theta_vec.mean(axis=0) < 0.9*np.pi)
        print(theta_vec/np.pi)
        self.assertAlmostEqual(correlation, 0.1875000)
        #runner.save_results("10thetas_16x16__", results)

    '''def test_generating_thetas(self):
        results = self.runner.optimize("avg_auto_corr", 10)
        self.assertTrue(len(results['results'])==10)
        with patch.object(self.runner, 'save_results') as save:
            save.side_effect = save_test_data
            self.runner.save_results("10thetas_4x4__", results)'''

    @show_plot(wait=3)
    def test_visualizing_polar_angles(self):
        theta_collections = extract_thetas_records("testdata/", "10thetas_4x4__today.json")
        for theta in theta_collections.thetas:
            polar_plot_angles(theta)

    @show_plot(wait=1, close=False)
    def test_visualizing_angles(self):
        theta_collections = extract_thetas_records("testdata/", "10thetas_4x4__today.json")
        for theta in theta_collections.thetas:
            plot_angles(theta)


    def tearDown(self):
        del self.runner


class TestWithMediumSize(unittest.TestCase):
    '''Testing gdft builder with 8x8 matrices'''
    def setUp(self):
        self.runner = Runner(8)

    def test_generating_one_gdft(self):
        results = self.runner.optimize("avg_auto_corr", 1)
        records = results['results']
        self.assertTrue(records is not [])
        theta_vec = records[0]['theta_vec']
        correlation = records[0]['correlation']
        self.assertTrue(0.1*np.pi < theta_vec.mean(axis=0) < 0.9*np.pi)
        print(theta_vec/np.pi)
        self.assertAlmostEqual(correlation, 0.1875000)
        #runner.save_results("10thetas_16x16__", results)

    '''def test_generating_thetas(self):
        results = self.runner.optimize("avg_auto_corr", 10)
        self.assertTrue(len(results['results'])==10)
        with patch.object(self.runner, 'save_results') as save:
            save.side_effect = save_test_data
            self.runner.save_results("10thetas_4x4__", results)'''

    @show_plot(wait=3)
    def test_visualizing_polar_angles(self, wait=3):
        theta_collections = extract_thetas_records("testdata/", "10thetas_4x4__today.json")
        for theta in theta_collections.thetas:
            polar_plot_angles(theta)

    @show_plot(wait=1, close=False)
    def test_visualizing_angles(self):
        theta_collections = extract_thetas_records("testdata/", "10thetas_4x4__today.json")
        for theta in theta_collections.thetas:
            plot_angles(theta)


    def tearDown(self):
        del self.runner

if __name__ == '__main__':
    unittest.main()
