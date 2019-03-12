from copy import deepcopy
import numpy as np
from scipy import linalg
from utils import show
from gdft import dft_matrix, gdft_matrix, two_param_gdft_matrix
from analyzer import ThetasAnalyzer
from correlations import CorrelationAnalyzer, Correlation


def ct_derivative(gdft, sigma, alpha, beta, pos_mu): #OK
    N = gdft.shape[0]
    mu = pos_mu - N + 1

    is_positive = 0 < mu <= N - 1
    is_negative = 1 - N <= mu <= 0

    def pos_val(sigma):
        res = 0
        if sigma <= N-1-mu:
            res += 1j*gdft[alpha, sigma] * np.conjugate(gdft[beta, sigma + mu])
        if sigma >= mu:
            res -= 1j*gdft[alpha, sigma - mu] * np.conjugate(gdft[beta, sigma])
        return res/N

    def neg_val(sigma):
        res = 0
        if sigma >= -mu:
            res += 1j*gdft[alpha, sigma] * np.conjugate(gdft[beta, sigma + mu])
        if sigma <= N - 1 + mu:
            res -= 1j*gdft[alpha, sigma - mu] * np.conjugate(gdft[beta, sigma])
        return res/N

    if is_positive:
        return pos_val(sigma)

    if is_negative:
        return neg_val(sigma)


def ct_gradient(gdft, alpha, beta, mu): #OK
    N = gdft.shape[0]
    derivatives = [ct_derivative(gdft, sigma, alpha, beta, mu) for sigma in range(N)]
    return np.array(derivatives)


def auto_corr_derivative(sigma, gdft): #OK
    N = gdft.shape[0]
    result = 0
    corr_tensor = Correlation(gdft).correlation_tensor()
    conj_tensor = np.conjugate(corr_tensor)
    for alpha in range(N):
        for mu in range(1, N):
            first_term = ct_derivative(gdft, sigma, alpha, alpha, mu) * conj_tensor[alpha, alpha, mu]
            second_term = np.conjugate(ct_derivative(gdft, sigma, alpha, alpha, mu)) * corr_tensor[alpha, alpha, mu]
            result += first_term + second_term
            # print(alpha, mu, result)
    return 2*result / N


def auto_corr_gradient(gdft): #OK
    N = gdft.shape[0]
    derivatives = [auto_corr_derivative(sigma, gdft) for sigma in range(N)]
    return np.array(derivatives)


def corr_difference(analyzer, theta, sigma, corr_name, h=0.00001): #OK
    N = theta.shape[0]
    # print(sigma, h)
    old_gdft = gdft_matrix(N, theta)
    old_corr_tensor = Correlation(old_gdft).correlation_tensor()
    old_corr = analyzer._corr_fns[corr_name](old_corr_tensor)
    new_theta = deepcopy(theta)
    new_theta[sigma] += h
    new_gdft = gdft_matrix(N, new_theta)
    new_corr_tensor = Correlation(new_gdft).correlation_tensor()
    new_corr = analyzer._corr_fns[corr_name](new_corr_tensor)
    return (new_corr - old_corr) / h


def corr_differences(analyzer, theta, corr_name, step=0.00001): #OK
    diffs = [corr_difference(analyzer, theta, index, corr_name, h=step) for index, _ in enumerate(theta)]
    return np.array(diffs)


def ct_difference(theta, sigma, h=0.00001):
    N = theta.shape[0]
    # print(sigma, h)
    old_gdft = gdft_matrix(N, theta)
    old_corr_tensor = Correlation(old_gdft).correlation_tensor()

    new_theta = deepcopy(theta)
    new_theta[sigma] += h
    new_gdft = gdft_matrix(N, new_theta)
    new_corr_tensor = Correlation(new_gdft).correlation_tensor()
    return (new_corr_tensor - old_corr_tensor) / h


def avg_corr_difference(theta, sigma, h=0.00001):
    N = theta.shape[0]
    ct_diff = ct_difference(theta, sigma, h=h)
    conj_ct_diff = np.conjugate(ct_diff)
    gdft = gdft_matrix(N, theta)
    ct = Correlation(gdft).correlation_tensor()

    def compute_term(a, mu):
        return ct_diff[a, a, mu] * np.conjugate(ct[a, a, mu]) + ct[a, a, mu] * conj_ct_diff[a, a, mu]

    result = 0
    for alpha in range(N):
        for mu in range(N - 1):
            result += compute_term(alpha, mu)
        for mu in range(N, 2 * N - 1):
            result += compute_term(alpha, mu)

    return result / N


def avg_corr_derivative(theta, sigma, h=0.00001):
    N = theta.shape[0]
    ct_diff = ct_difference(theta, sigma, h=h)
    conj_ct_diff = np.conjugate(ct_diff)
    gdft = gdft_matrix(N, theta)
    ct = Correlation(gdft).correlation_tensor()

    def compute_term(a, mu):
        return ct_diff[a, a, mu] * np.conjugate(ct[a, a, mu]) + ct[a, a, mu] * conj_ct_diff[a, a, mu]

    result = 0
    for alpha in range(N):
        for mu in range(N - 1):
            result += compute_term(alpha, mu)
        for mu in range(N, 2 * N - 1):
            result += compute_term(alpha, mu)

    return result / N