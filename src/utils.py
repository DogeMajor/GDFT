import time

def timer(function):
    def timer_wrapper(*args, **kwargs):
        t0 = time.time()
        result = function(*args,**kwargs)
        delta_t = time.time()-t0
        print('Function {} took {} seconds to run.'.format(function.__name__, delta_t))
        return result
    return timer_wrapper
    #Defines a timer decorator

def show(function):
    def inner_fn(self, *args, **kwargs):
        fn_name = function.__name__
        cl_name = self.__class__.__name__
        print("Using {0} of class {1}".format(fn_name, cl_name))
        return function(self, *args, **kwargs)
    return inner_fn