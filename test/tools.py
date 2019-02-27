import matplotlib.pyplot as plt
import numpy as np


def EqualMatrices(matA, matB):
    if matA.shape != matB.shape:
        return False
    return (matA == matB).all()


def AssertAlmostEqualMatrices(matA, matB, decimals=7):
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
