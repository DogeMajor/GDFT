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

#----mocked methods/fns/classes--------

def save_test_data(file_name, results):
    date_string = "today"
    dao = DAO("testdata/")
    dao.write(file_name + date_string + ".json", results)

class TestWithSmallSize(unittest.TestCase):
    #Testing gdft builder with 4x4 matrices
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

    def test_generating_thetas(self):
        results = self.runner.optimize("avg_auto_corr", 10)
        self.assertTrue(len(results['results'])==10)
        with patch.object(self.runner, 'save_results') as save:
            save.side_effect = save_test_data
            self.runner.save_results("10thetas_4x4__", results)

    @show_plot(wait=3)
    def test_visualizing_polar_angles(self):
        theta_collections = extract_thetas_records("testdata/", "10thetas_4x4__today.json")
        for theta in theta_collections.thetas:
            polar_plot_angles(theta)

    @show_plot(wait=1, close=False)
    def test_visualizing_angles(self):
        theta_collections = extract_thetas_records("testdata/", "10thetas_4x4__today.json")
        print(theta_collections.thetas)
        for theta in theta_collections.thetas:
            plot_angles(sorted(theta))


    def tearDown(self):
        del self.runner


'''class TestWithMediumSize(unittest.TestCase):
    #Testing gdft builder with 8x8 matrices
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
        print(results)
        self.assertTrue(correlation < 0.13)

    def test_generating_thetas(self):
        results = self.runner.optimize("avg_auto_corr", 10)
        self.assertTrue(len(results['results'])==10)
        with patch.object(self.runner, 'save_results') as save:
            save.side_effect = save_test_data
            self.runner.save_results("10thetas_8x8__", results)

    @show_plot(wait=3)
    def test_visualizing_polar_angles(self, wait=3):
        theta_collections = extract_thetas_records("testdata/", "10thetas_8x8__today.json")
        for theta in theta_collections.thetas:
            polar_plot_angles(theta)

    @show_plot(wait=1, close=False)
    def test_visualizing_angles(self):
        theta_collections = extract_thetas_records("testdata/", "10thetas_8x8__today.json")
        for theta in theta_collections.thetas:
            plot_angles(theta)


    def tearDown(self):
        del self.runner



class TestWithLargeSize(unittest.TestCase):
    #Testing gdft builder with 16x16 matrices
    def setUp(self):
        self.runner = Runner(16)

    def test_generating_one_gdft(self):
        results = self.runner.optimize("avg_auto_corr", 1)
        records = results['results']
        self.assertTrue(records is not [])
        theta_vec = records[0]['theta_vec']
        correlation = records[0]['correlation']
        self.assertTrue(0.1*np.pi < theta_vec.mean(axis=0) < 0.9*np.pi)
        print(theta_vec/np.pi)
        print(results)
        self.assertTrue(correlation < 0.13)

    def test_generating_thetas(self):
        results = self.runner.optimize("avg_auto_corr", 10)
        self.assertTrue(len(results['results'])==10)
        with patch.object(self.runner, 'save_results') as save:
            save.side_effect = save_test_data
            self.runner.save_results("10thetas_16x16__", results)

    @show_plot(wait=3)
    def test_visualizing_polar_angles(self, wait=3):
        theta_collections = extract_thetas_records("testdata/", "10thetas_16x16__today.json")
        for theta in theta_collections.thetas:
            polar_plot_angles(theta)

    @show_plot(wait=1, close=False)
    def test_visualizing_angles(self):
        theta_collections = extract_thetas_records("testdata/", "10thetas_16x16__today.json")
        for theta in theta_collections.thetas:
            plot_angles(theta)


    def tearDown(self):
        del self.runner'''

if __name__ == '__main__':
    unittest.main()
