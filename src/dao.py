import json
import codecs
import numpy as np


def complex_decoder(obj):
    data = obj["ComplexNumber"]
    return data["Re"] + 1j * data["Im"]

def complex_array_decoder(obj):
    data = obj["ComplexArray"]
    real_array = np.array(data["Re"])
    imag_array = np.array(data["Im"])
    return real_array + 1j * imag_array


def complex_encoder(number):
    return {"ComplexNumber": {"Re": number.real, "Im": number.imag}}

def complex_array_encoder(array):
    return {"ComplexArray": {"Re": list(array.real), "Im": list(array.imag)}}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return complex_encoder(obj)

        if isinstance(obj,(np.ndarray,)) and np.iscomplexobj(obj):
            print("found complex array ", obj)
            return complex_array_encoder(obj)

        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()

        return json.JSONEncoder.default(self, obj)

class DAO(object):

    def __init__(self, path):
        self._path = path

    def read(self, file_name, path=None):
        if path == None:
            path = self._path
        try:
            with codecs.open(path+file_name, 'r', 'utf-8') as file_object:
                return json.loads(file_object.read())
        except IOError:
            print('Can\'t read the file called {}'.format(path+file_name))


    def write(self, file_name, content, path=None):
        if path == None:
            path = self._path
        try:
            with open(path+file_name, 'w') as file_object:
                #content = to_json_obj(content)
                json.dump(content, file_object, ensure_ascii=False, cls=NumpyEncoder)
        except IOError:
            print('Can\'t write the file {}'.format(path+file_name))
