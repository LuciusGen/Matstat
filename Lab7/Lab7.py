import numpy as np
from tabulate import tabulate
import scipy.stats as stats
import sys

alpha = 0.05
p = 1 - alpha
k = 6

sys.stdout = open("out.txt", "w")

dist = np.random.normal(0, 1, size=100)
mu = np.mean(dist)
sigma = np.std(dist)

print(np.around(mu, decimals=2), ' ', np.around(sigma, decimals=2))
print()

delta = np.linspace(-1.01, 1.56, num=k - 1)
value = stats.chi2.ppf(p, k - 1)
array = np.array([stats.norm.cdf(delta[0])])
q = np.array([len(dist[dist <= delta[0]])])

for i in range(0, len(delta) - 1):
    new_ar = stats.norm.cdf(delta[i + 1]) - stats.norm.cdf(delta[i])
    array = np.append(array, new_ar)
    q = np.append(q, len(dist[(dist <= delta[i + 1]) & (dist >= delta[i])]))


array = np.append(array, 1 - stats.norm.cdf(delta[-1]))
q = np.append(q, len(dist[dist >= delta[-1]]))
result = np.divide(np.multiply((q - 100 * array), (q - 100 * array)), array * 100)

table_headers = ["i", '$Delta_i$', "$n_i$", "$p_i$", "$np_i$", "$n_i-np_i$", "$frac{(n_i-np_i)^2}{np_i}$"]
rows = []

for i in range(0, len(q)):
    if i == 0:
        boarders = ['-infity', np.around(delta[0], decimals=2)]
    elif i == len(q) - 1:
        boarders = [np.around(delta[-1], decimals=2), 'infity']
    else:
        boarders = [np.around(delta[i - 1], decimals=2), np.around(delta[i], decimals=2)]

    rows.append([i + 1, boarders, q[i], np.around(array[i], decimals=4), np.around(array[i] * 100, decimals=2),
                 np.around(q[i] - 100 * array[i], decimals=2), np.around(result[i], decimals=2)])


rows.append([len(q), "-", np.sum(q), np.around(np.sum(array), decimals=4),
             np.around(np.sum(array * 100), decimals=2),
             -np.around(np.sum(q - 100 * array), decimals=2),
             np.around(np.sum(result), decimals=2)])


print(tabulate(rows, table_headers, tablefmt="latex"))
