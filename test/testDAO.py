import sys
sys.path.append("../src")
sys.path.append("src/")
import os
import unittest
import time
import json
import numpy as np
from tools import *
from dao import *
np.random.seed(int(time.time()))


class TestDAO(unittest.TestCase):

    '''def test_write_and_read(self):
        dao = DAO("../data/")
        balance = int(np.random.randint(1, 10000))
        content = {"name": "Julgubbe",
                   "balance": balance}
        dao.write("julinfo.json", content, "../data/")
        retrieved_content = dao.read("julinfo.json")
        self.assertEqual(retrieved_content['balance'], balance)
        os.remove("../data/julinfo.json")
        cont = dao.read("julinfo.json")
        self.assertEqual(cont, None)

    def test_read(self):
        dao = DAO("../data/")
        retrieved_content = dao.read("static_julinfo.json")
        self.assertEqual(retrieved_content['balance'], 1000)
        self.assertEqual(retrieved_content['name'], "Jultomte")

    def test_saving_int_numpy_matrix(self):
        dao = DAO("../data/")
        matrix = np.array([1,2,3], dtype=np.int32)
        content = {"name": "random matrix",
                   "matrix": matrix}
        #content = {"name": "random matrix",
        #           "matrix": [1,2,3]}
        print(type(matrix))
        dao.write("matrix_info.json", content, "../data/")
        retrieved_content = dao.read("matrix_info.json")
        self.assertEqual(retrieved_content['matrix'], [1, 2, 3])
        os.remove("../data/matrix_info.json")

    def test_complex_encodings(self):
        encoder = NumpyEncoder()
        matrix = np.array([1 + 1j, 2 - 2 * 1j, 3 - 3 * 1j])
        json_array = encoder.complex_array_encoder(matrix)
        self.assertEqual(json_array["ComplexArray"]["Re"], [1, 2, 3])
        self.assertEqual(json_array["ComplexArray"]["Im"], [1, -2, -3])
        complex_json = encoder.complex_number_encoder(1 - 1j)
        self.assertEqual(complex_json["ComplexNumber"]["Re"], 1)
        self.assertEqual(complex_json["ComplexNumber"]["Im"], -1)

    def test_complex_decodings(self):
        decoder = ComplexDecoder()

        complex_json = {"ComplexNumber": {"Re": 1, "Im": -1}}
        self.assertEqual(decoder.complex_number_decoder(complex_json), 1 - 1j)

        complex_mat_json = {"ComplexArray": {"Re": [1, 2, 3], "Im": [1, -2, -3]}}

        matrix = decoder.complex_array_decoder(complex_mat_json)
        self.assertEqual(list(matrix),
                         list(np.array([1 + 1j, 2 - 2 * 1j, 3 - 3 * 1j])))


    def test_saving_complex_numpy_vector(self):
        dao = DAO("../data/")
        vector = np.array([1+1j, 2-2*1j, 3-3*1j])
        content = {"name": "complex vector",
                   "vector": vector}

        dao.write("vector_info.json", content, "../data/")
        retrieved_content = dao.read("vector_info.json")
        retrieved_vec = retrieved_content['vector']
        self.assertEqual(list(retrieved_vec), list(vector))
        os.remove("../data/vector_info.json")'''


    def test_saving_complex_numpy_matrix(self):
        dao = DAO("../data/")
        matrix = np.array([[1+1j, 2-2*1j], [3-3*1j, 4-4*1j]])
        content = {"name": "complex matrix",
                   "matrix": matrix}

        dao.write("matrix_info.json", content, "../data/")
        retrieved_content = dao.read("matrix_info.json")
        retrieved_mat = retrieved_content['matrix']
        print(retrieved_mat)
        self.assertTrue(EqualMatrices(retrieved_mat, matrix))
        os.remove("../data/matrix_info.json")