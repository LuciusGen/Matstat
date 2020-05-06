import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sys

sys.stdout = open("data.txt", "w")


def fit(X, params):
    return X.dot(params)


def cost_function(params, X, y):
    return np.sum(np.abs(y - fit(X, params)))


def generate(a, b, h):
    n = int((b - a) / h) + 2
    x = np.linspace(-1.8, 2, n)
    e_i = np.random.standard_normal(n)
    y = 2 + 2*x + e_i

    return x, y


def perturbation(y):
    res = []
    for i in range(0, len(y)):
        res.append(y[i])
    back = len(y) - 1
    res[0] = y[0] + 10
    res[back] = y[back] - 10

    return res


def mnk(x, y):
    a2 = (np.mean(x*y) - np.mean(x) * np.mean(y))/(np.mean(x*x) - np.mean(x)**2)
    b2 = np.mean(y) - a2 * np.mean(x)

    return a2, b2


def mnm(x, y, betas):
    X = np.asarray([np.ones(20, ), x]).T
    output = minimize(cost_function, betas, args=(X, y), method='SLSQP')

    y_hat = fit(X, output.x)

    return x, y_hat, output.x


l = -1.8
r = 2
h = 0.2
a = 2
b = 2

plt.figure()
plt.subplot(121)
plt.title("Оригинал")
print("\t\t\tOriginal")

x, y = generate(l, r, h)
print("%12s:\t a = %lf, b = %lf" % ("Model sample", a, b))
plt.plot(x, a * x + b, 'b', label='Модель')

plt.scatter(x, y)

m, c = mnk(x, y)
print("%12s:\ta = %lf, b = %lf" % ("МНК", m, c))
plt.plot(x, m*x + c, 'r', label='МНаимКвадратов')

m, c, k = mnm(x, y, np.array([2, 2]))
plt.plot(m, c, 'g', label='МНаимМодулей')
print("%12s:\ta = %lf, b = %lf" % ("МНМ", k[0], k[1]))
plt.legend()

print("\n")
plt.subplot(122)
plt.title("С погрешностью")
print("\t\t\tDistorted sample")

x, y = generate(l, r, h)
y = perturbation(y)

print("%12s:\ta = %lf, b = %lf" % ("Model sample", a, b))
plt.plot(x, a * x + b, 'b', label='Модель')
plt.scatter(x, y)

m, c = mnk(x, y)
plt.plot(x, m*x + c, 'r', label='МНаимКвадратов')
print("%12s:\ta = %lf, b = %lf" % ("МНК", m, c))

xx, yy, k = mnm(x, y, np.array([2, 2]))
plt.plot(xx, yy, 'g', label='МНаимМодулей')
print("%12s:\ta = %lf, b = %lf" % ("МНМ", k[0], k[1]))

plt.legend()
plt.savefig("Graph")
plt.close()
sys.stdout.close()
