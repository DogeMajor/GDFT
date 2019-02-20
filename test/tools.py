import matplotlib.pyplot as plt
import numpy as np


def EqualMatrices(matA, matB):
    are_equal = True
    if matA.shape != matB.shape:
        return False
    return (matA==matB).all()
    '''for row in range(matA.shape[0]):
        for col in range(matA.shape[1]):
            are_equal = are_equal and bool(matA[row, col] == matB[row, col])
    return are_equal'''


def AssertAlmostEqualMatrices(matA, matB, decimals=7):
    are_equal = True
    if matA.shape != matB.shape:
        return False
    return np.testing.assert_almost_equal(matA, matB, decimal=decimals)


def show_plot(wait=3, close=True):

    def _show_plot(fn):
        title = fn.__name__

        def plot_fn(self, *args, **kwargs):
            fn(self, *args, **kwargs)
            plt.title(title)
            plt.pause(wait)
            if close:
                plt.close()
            plt.show()
        return plot_fn
    return _show_plot
