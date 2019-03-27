import time
import datetime
from collections import namedtuple
from math import sqrt
import numpy as np
from dao import ThetasDAO

'''General functions / decorator to help the dev process and
to take care of file handling + administrative processes.'''


def timer(function):
    def timer_wrapper(*args, **kwargs):
        start_time = time.clock()
        result = function(*args, **kwargs)
        delta_t = time.clock() - start_time
        print('Function {} took {} seconds to run.'.format(function.__name__, delta_t))
        return result
    return timer_wrapper


def show(function):
    def inner_fn(self, *args, **kwargs):
        fn_name = function.__name__
        cl_name = self.__class__.__name__
        print("Using {0} of class {1}".format(fn_name, cl_name))
        return function(self, *args, **kwargs)
    return inner_fn


def datetime_encoder(obj):
    if isinstance(obj, datetime.datetime):
        return "{}-{}_{}_{}".format(obj.month, obj.day, obj.hour, obj.minute)
    return obj


def save_as_json(function):
    def inner_fn(self, *args, **kwargs):
        results = function(self, *args, **kwargs)
        print("Saving results to a json file")
        date_string = datetime_encoder(datetime.datetime.now())
        file_name = "results_"+date_string+".json"
        dao = ThetasDAO("../data/")
        dao.write(file_name, results)
        return results
    return inner_fn


Thetas = namedtuple('Thetas', 'thetas correlations')


def extract_thetas_records(path, file_name):
    dao = ThetasDAO(path)
    content = dao.read(file_name)
    theta_vecs = [np.array(result["theta_vec"]) for result in content["results"]]
    corrs = [result["correlation"] for result in content["results"]]
    return Thetas(thetas=theta_vecs, correlations=corrs)

def seq_norm(seq_a, seq_b):
    distances = ((item_b - item_a) ** 2 for item_a, item_b in zip(seq_a, seq_b))
    return sqrt(sum(distances))


def to_phase(matrix):
    return np.angle(matrix)

def small_els_to(unitary_matrix, replace_val=0, cutoff=0.1):
    unitary_matrix[np.absolute(unitary_matrix) < cutoff] = replace_val
    return unitary_matrix

def big_els_to(unitary_matrix, replace_val=1, cutoff=0.1):
    unitary_matrix[np.absolute(unitary_matrix-replace_val) < cutoff] = replace_val
    return unitary_matrix

def approximate_matrix(unitary_matrix, tol=0.1):
    unitary_matrix = small_els_to(unitary_matrix, replace_val=0, cutoff=tol)
    return big_els_to(unitary_matrix, replace_val=1, cutoff=tol)

def approximate_phases(matrix, tol=0.1*np.pi):
    phases = to_phase(matrix)
    phases = small_els_to(phases, replace_val=0, cutoff=tol)
    phases = big_els_to(phases, replace_val=2*np.pi, cutoff=tol)
    np.place(phases, phases == 2*np.pi, 0)
    return phases
