import time
import datetime
from collections import namedtuple
from dao import *

def timer(function):
    def timer_wrapper(*args, **kwargs):
        t0 = time.clock()
        result = function(*args, **kwargs)
        delta_t = time.clock()-t0
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
        dao = DAO("../data/")
        dao.write(file_name, results)
        return results
    return inner_fn


Thetas = namedtuple('Thetas', 'thetas correlations')

def extract_thetas_records(path, file_name):
    dao = DAO(path)
    content = dao.read(file_name)
    theta_vecs = [np.array(result["theta_vec"]) for result in content["results"]]
    corrs = [result["correlation"] for result in content["results"]]
    return Thetas(thetas=theta_vecs, correlations=corrs)
