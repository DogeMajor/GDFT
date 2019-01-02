
class SequenceFinder(object):

    def __init__(self):
        pass

    def _to_integers(self, seq, scaling=1):
        abs_seq = [abs(item) for item in seq]
        min_val = min(abs_seq)
        def _to_int(item):
            return int((scaling/min_val) * item)
        return list(map(_to_int, seq))

    def _to_abs_integers(self, seq, scaling=1):
        abs_seq = [abs(item) for item in seq]
        return self._to_integers(abs_seq, scaling)

    def nth_diff(self, seq, n):
        '''Forward difference'''
        def _diff(ind):
            return seq[ind+1] - seq[ind]

        length = len(seq)
        if length <= 1:
            return []
        elif n == 0:
            return seq
        elif n == 1:
            return [_diff(ind) for ind in range(length - 1)]

        seq = self.nth_diff(seq, 1)
        return self.nth_diff(seq, n - 1)


class FirstDifference(object):

    def __init__(self, seq):
        self._seq = seq

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        try:
            res = self._seq[self._index + 1] - self._seq[self._index]
            self._index += 1
            return res
        except IndexError:
            raise StopIteration




diff = iter(FirstDifference([1,4,9,16]))

#for i in range(7):
#    print(next(diff))

for val in diff:
    print(val)



