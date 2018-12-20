import json
import codecs
import numpy as np


class ComplexDecoder(object):
    '''Decodes json complex arrays to lists (or lists of lists etc.), not numpy.arrays.
    Using np.array(result) will make np.arrays from these!!
    Shall be used as hook for read in DAO's json.loads.'''

    def complex_number_decoder(self, obj):
        data = obj["ComplexNumber"]
        return data["Re"] + 1j * data["Im"]

    def complex_array_decoder(self, obj):
        data = obj["ComplexArray"]
        real_array = np.array(data["Re"])
        imag_array = np.array(data["Im"])
        return real_array + 1j * imag_array

    def decode(self, obj):
        if "ComplexNumber" in obj:
            return self.complex_number_decoder(obj)

        elif "ComplexArray" in obj:
            return self.complex_array_decoder(obj)
        return obj


class NumpyEncoder(json.JSONEncoder):
    '''Encodes normal np objects to numbers or lists, whereas for
    complex numbers and arrays special json formats will be used.'''

    def complex_number_encoder(self, number):
        return {"ComplexNumber": {"Re": number.real, "Im": number.imag}}

    def complex_array_encoder(self, array):
        return {"ComplexArray": {"Re": list(array.real), "Im": list(array.imag)}}


    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return self.complex_number_encoder(obj)

        if isinstance(obj, (np.ndarray,)) and np.iscomplexobj(obj):
            return self.complex_array_encoder(obj)

        elif isinstance(obj, (np.ndarray,)):
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
                decoder = ComplexDecoder()
                raw_data = file_object.read()
                return json.loads(raw_data, object_hook=decoder.decode)
        except IOError:
            print('Can\'t read the file called {}'.format(path+file_name))


    def write(self, file_name, content, path=None):
        if path == None:
            path = self._path
        try:
            with open(path+file_name, 'w') as file_object:
                json.dump(content, file_object, ensure_ascii=False, cls=NumpyEncoder)
        except IOError:
            print('Can\'t write the file {}'.format(path+file_name))
