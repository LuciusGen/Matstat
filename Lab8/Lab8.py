import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys

sys.stdout = open("out.txt", "w")


def m_interval(sample):
    m = np.mean(sample)
    s = np.std(sample)
    n = len(sample)
    interval = s * stats.t.ppf((1 + gamma) / 2, n - 1) / (n - 1) ** 0.5
    return np.around(m - interval, decimals=2), np.around(m + interval, decimals=2)


def var_interval(sample):
    s = np.std(sample)
    n = len(sample)
    low = s * (n / stats.chi2.ppf((1 + gamma) / 2, n - 1)) ** 0.5
    high = s * (n / stats.chi2.ppf((1 - gamma) / 2, n - 1)) ** 0.5
    return np.around(low, decimals=2), np.around(high, decimals=2)


def m_asimpt(sample):
    m = np.mean(sample)
    s = np.std(sample)
    n = len(sample)
    u = stats.norm.ppf((1 + gamma) / 2)
    interval = s * u / (n ** 0.5)
    return np.around(m - interval, decimals=2), np.around(m + interval, decimals=2)


def var_asimpt(sample):
    s = np.std(sample)
    n = len(sample)
    m_4 = stats.moment(sample, 4)
    e = m_4 / s**4 - 3
    u = stats.norm.ppf((1 + gamma) / 2)
    U = u * (((e + 2) / n) ** 0.5)
    low = s * (1 + 0.5 * U) ** (-0.5)
    high = s * (1 - 0.5 * U) ** (-0.5)
    return np.around(low, decimals=2), np.around(high, decimals=2)


samples = [20, 100]
gamma = 0.95

for num in samples:
    dist = np.random.normal(0, 1, size=num)

    print('n = ' + str(num))
    print('mean', m_interval(dist))
    print('var', var_interval(dist))
    print()
    print("---------------------------------------------")
    print()
    print('size = ' + str(num))
    print('assimp_mean', m_asimpt(dist))
    print('assimp_var', var_asimpt(dist))
    print()
    print("---------------------------------------------")
    print()

sys.stdout.close()
