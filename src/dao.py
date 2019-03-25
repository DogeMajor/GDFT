import json
import csv
import codecs
from collections import Counter
import numpy as np
from analyzer import SortedThetas


class ComplexDecoder(object):
    '''Decodes json complex arrays to lists (or lists of lists etc.), not numpy.arrays.
    Using np.array(result) will make np.arrays from these. There is no need to decode
    all lists to numpy matrices, since that would require making all np.arrays
    into special objects ala complex_array_json (below)!!
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

        if "ComplexArray" in obj:
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



class BaseDAO(object):
    '''Writes json files of records in numpy format and reads
    jsons into numpy/list formats'''

    def __init__(self, path):
        self._path = path

    def read(self, file_name, path=None):
        if path is None:
            path = self._path
        try:
            with codecs.open(path+file_name, 'r', 'utf-8') as file_object:
                return self._read(file_object)
        except IOError:
            print('Can\'t read the file called {}'.format(path+file_name))
            raise

    def _read(self, file_object):
        return file_object.read()

    def write(self, file_name, content, path=None):
        if path is None:
            path = self._path
        try:
            with open(path+file_name, 'w') as file_object:
                self._write(file_object, content)
        except IOError:
            print('Can\'t write the file {}'.format(path+file_name))
            raise

    def _write(self, file_object, content):
        return file_object.write(content)


class ThetasDAO(BaseDAO):

    def __init__(self, path):
        BaseDAO.__init__(self, path)

    def _read(self, file_object):
        decoder = ComplexDecoder()
        raw_data = file_object.read()
        return json.loads(raw_data, object_hook=decoder.decode)

    def _write(self, file_object, content):
        json.dump(content, file_object, ensure_ascii=False, cls=NumpyEncoder)


class SortedThetasDAO(BaseDAO):

    def __init__(self, path):
        BaseDAO.__init__(self, path)

    def _to_histogram(self, groups):
        amounts = {key: len(value) for key, value in groups.items()}
        return Counter(amounts)

    def _get_headers(self, dim):
        return ("theta_" + str(index) for index in range(dim))

    def _read(self, file_object):
        reader = csv.reader(file_object, delimiter=',')
        headers = next(reader)
        thetas = {}
        labels = []
        for row in reader:
            if len(row) != 0:
                if row[0] is 'average':
                    labels.append(np.array(row[1:], dtype=np.float64))
                elif row[0] not in thetas.keys():
                    thetas[row[0]] = []
                thetas[row[0]].append(np.array(row[1:], dtype=np.float64))
        del thetas['average']
        thetas = {int(key): value for key, value in thetas.items()}
        hist = self._to_histogram(thetas)
        return SortedThetas(thetas=thetas, labels=labels, histogram=hist)

    def _write(self, file_object, sorted_thetas):
        dim = sorted_thetas.thetas[0][0].shape[0]
        headers = list(self._get_headers(dim))
        fieldnames = ['Label'] + headers
        thetas_writer = csv.writer(file_object, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        thetas_writer.writerow(fieldnames)
        for label_ind in sorted_thetas.thetas.keys():
            for theta in sorted_thetas.thetas[label_ind]:
                thetas_writer.writerow([label_ind] + theta.tolist())
            thetas_writer.writerow(['average'] + sorted_thetas.labels[label_ind].tolist())



class AnalyzedThetasDAO(BaseDAO):
    '''Writes csv files of the analysis of the records and reads them'''

    def __init__(self, path):
        BaseDAO.__init__(self, path)

    def _to_csv(self, json):
        pass

    def _read(self, file_object):
        #csv_data = csv.DictReader(file_object)
        #column_names = ", ".join(csv_data[0])
        reader = csv.reader(file_object, delimiter=',', quotechar='|')
        headers = next(reader)
        print(headers)
        #analyzed_thetas = {header: }
        for index, row in reader:
            print(index, row)

    def _write(self, file_object, content):
        fieldnames = ['label_index', 'last_name']
        writer = csv.DictWriter(file_object, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'first_name': 'Baked', 'last_name': 'Beans'})
        writer.writerow({'first_name': 'Lovely', 'last_name': 'Spam'})
        writer.writerow({'first_name': 'Wonderful', 'last_name': 'Spam'})
