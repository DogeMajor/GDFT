import time
import datetime
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
        return "{}-{}-{} {}-{}".format(obj.year, obj.month, obj.day, obj.hour, obj.minute)
    return obj

def save_as_json(function):
    def inner_fn(self, *args, **kwargs):
        results = function(self, *args, **kwargs)
        print("Saving results to a json file")
        file_name = "results_"+str(time)+".json"
        dao = DAO("../data/")
        dao.write(results, results)
        return results
    return inner_fn
