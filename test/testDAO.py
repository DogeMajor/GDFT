import os
import unittest
import time
import sys
import numpy as np
sys.path.append("../src")
sys.path.append("src/")
from tools import GDFTTestCase
from dao import BaseDAO, ThetasDAO, SortedThetasDAO, AnalyzedThetasDAO, NumpyEncoder, ComplexDecoder
from analyzer import SortedThetas

PATH = "test/testdata/"
np.random.seed(int(time.time()))



class TestThetasDAO(unittest.TestCase):

    def test_write_and_read(self):
        dao = ThetasDAO(PATH)
        balance = int(np.random.randint(1, 10000))
        content = {"name": "Julgubbe",
                   "balance": balance}
        dao.write("julinfo.json", content, PATH)
        retrieved_content = dao.read("julinfo.json")
        self.assertEqual(retrieved_content['balance'], balance)
        os.remove(PATH+"julinfo.json")
        self.assertTrue('julinfo.json' not in os.listdir(PATH))

    def test_read(self):
        dao = ThetasDAO(PATH)
        retrieved_content = dao.read("static_julinfo.json")
        self.assertEqual(retrieved_content['balance'], 1000)
        self.assertEqual(retrieved_content['name'], "Jultomte")

    def test_saving_int_numpy_vector(self):
        dao = ThetasDAO(PATH)
        vector = np.array([1, 2, 3], dtype=np.int32)
        content = {"name": "int_vector",
                   "vector": vector}

        dao.write("int_vector_info.json", content, PATH)
        retrieved_content = dao.read("int_vector_info.json")
        self.assertEqual(retrieved_content['vector'], [1, 2, 3])
        os.remove(PATH+"int_vector_info.json")

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
        dao = ThetasDAO(PATH)
        vector = np.array([1+1j, 2-2*1j, 3-3*1j])
        content = {"name": "complex vector",
                   "vector": vector}

        dao.write("vector_info.json", content, PATH)
        retrieved_content = dao.read("vector_info.json")
        retrieved_vec = retrieved_content['vector']
        self.assertEqual(list(retrieved_vec), list(vector))
        os.remove(PATH+"vector_info.json")

    def test_saving_complex_numpy_matrix(self):
        dao = ThetasDAO(PATH)
        matrix = np.array([[1+1j, 2-2*1j], [3-3*1j, 4-4*1j]])
        content = {"name": "complex matrix",
                   "matrix": matrix}

        dao.write("matrix_info.json", content, PATH)
        retrieved_content = dao.read("matrix_info.json")
        retrieved_mat = retrieved_content['matrix']
        self.assertTrue(EqualMatrices(retrieved_mat, matrix))
        os.remove(PATH+"matrix_info.json")




FIRST_THETAS_GROUP = [np.array([2.93467664, 0.3384844, 2.87214115, 1.20613475, 0.32252419,
                                0.22130658, 2.20358886, 3.03256426]),
                      np.array([2.9696509, 0.32219672, 2.80460713, 1.08736893, 0.15248202,
                                0., 1.93102203, 2.70875109]),
                      np.array([3.14150297, 0.50597848, 3.00032683, 1.29500184, 0.37205216,
                                0.23150659, 2.17446601, 2.96410783])]

SECOND_THETAS_GROUP = [np.array([0.23263316, 1.06778065, 3.05624654, 2.96119473,
                                 2.08375977, 0.4239405, 2.96378942, 0.37377238]),
                       np.array([0.12853125, 0.96144711, 2.94767889, 2.85038891,
                                 1.97072153, 0.30866741, 2.8462803, 0.25403236]),
                       np.array([0.43171271, 1.20998793, 3.14159265, 2.9896899,
                                 2.05536774, 0.33869695, 2.82167874, 0.17477883]),
                       np.array([0.4590644, 1.22344731, 3.14116473, 2.97536152,
                                 2.02716572, 0.29658784, 2.76567391, 0.10491548])]

SORTED_THETAS = SortedThetas(thetas={0: FIRST_THETAS_GROUP, 1: SECOND_THETAS_GROUP},
                             labels=[FIRST_THETAS_GROUP[0], SECOND_THETAS_GROUP[0]],
                             histogram={0: 3, 1:4})


class TestSortedThetasDAO(GDFTTestCase):

    def test_write_and_read(self):
        dao = SortedThetasDAO(PATH)
        dao.write("sorted_thetas.csv", SORTED_THETAS, PATH)
        retrieved_content = dao.read("sorted_thetas.csv")
        self.assertAlmostEqualMatrices(retrieved_content.thetas[0][0], SORTED_THETAS.thetas[0][0])
        self.assertAlmostEqualMatrices(retrieved_content.thetas[1][0], SORTED_THETAS.thetas[1][0])
        self.assertEqual(retrieved_content.histogram, SORTED_THETAS.histogram)
        os.remove(PATH+"sorted_thetas.csv")
        self.assertTrue("sorted_thetas.csv" not in os.listdir(PATH))


class TestAnalyzedThetasDAO(unittest.TestCase):

    def test_write_and_read(self): #False!!!!!!!
        #print(SORTED_THETAS.thetas[0][0].dtype)
        dao = SortedThetasDAO(PATH)
        dao.write("sorted_thetas.csv", SORTED_THETAS, PATH)
        retrieved_content = dao.read("sorted_thetas.csv")
        print(retrieved_content)
        os.remove(PATH+"sorted_thetas.csv")
        self.assertTrue("sorted_thetas.csv" not in os.listdir(PATH))

    def test_read(self):
        dao = AnalyzedThetasDAO(PATH)
        retrieved_content = dao.read("analyzed_thetas.csv")
        #self.assertEqual(retrieved_content['balance'], 1000)
        #self.assertEqual(retrieved_content['name'], "Jultomte")

    def test_saving_int_numpy_vector(self):
        dao = AnalyzedThetasDAO(PATH)
        vector = np.array([1, 2, 3], dtype=np.int32)
        content = {"name": "int_vector",
                   "vector": vector}

        dao.write("int_vector_info.json", content, PATH)
        retrieved_content = dao.read("int_vector_info.json")
        self.assertEqual(retrieved_content['vector'], [1, 2, 3])
        os.remove(PATH+"int_vector_info.json")

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
        dao = AnalyzedThetasDAO(PATH)
        vector = np.array([1+1j, 2-2*1j, 3-3*1j])
        content = {"name": "complex vector",
                   "vector": vector}

        dao.write("vector_info.json", content, PATH)
        retrieved_content = dao.read("vector_info.json")
        retrieved_vec = retrieved_content['vector']
        self.assertEqual(list(retrieved_vec), list(vector))
        os.remove(PATH+"vector_info.json")

    def test_saving_complex_numpy_matrix(self):
        dao = AnalyzedThetasDAO(PATH)
        matrix = np.array([[1+1j, 2-2*1j], [3-3*1j, 4-4*1j]])
        content = {"name": "complex matrix",
                   "matrix": matrix}

        dao.write("matrix_info.json", content, PATH)
        retrieved_content = dao.read("matrix_info.json")
        retrieved_mat = retrieved_content['matrix']
        self.assertTrue(EqualMatrices(retrieved_mat, matrix))
        os.remove(PATH+"matrix_info.json")

if __name__ == '__main__':
    unittest.main()