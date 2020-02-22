import numpy as np
import matplotlib.pyplot as plt




LAMBDA = 10#for poisson
BOUND = np.sqrt(3)#for uniform

def normalizedDistr(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-x * x / 2)

def laplaceDistr(x):
    return (1 / np.sqrt(2)) * np.exp(-np.sqrt(2) * np.abs(x))

def cauchyDistr(x):
    return 1 / (np.pi * (1 + x * x))

def poissonDistr(x):
    return (np.power(x, LAMBDA) / np.math.factorial(LAMBDA)) * np.exp(-x)

def uniformDistr(x):
    return 1 / (2 * BOUND) * (x <= BOUND)

def laplaceGen(x):
    return np.random.laplace(0, 1/np.sqrt(2), x)

def poissonGen(x):
    return np.random.poisson(LAMBDA, x)

def uniformGen(x):
    return np.random.uniform(-BOUND, BOUND, x)

distrs = {
    'normal'  : normalizedDistr,
    'laplace' : laplaceDistr,
    'cauchy'  : cauchyDistr,
    'poisson' : poissonDistr,
    'uniform' : uniformDistr,
}

generateDict = {
    'normal'  : np.random.standard_normal,
    'laplace' : laplaceGen,
    'cauchy'  : np.random.standard_cauchy,
    'poisson' : poissonGen,
    'uniform' : uniformGen,
}

def draw(array, func, chunk, i):
    plt.subplot(221 + chunk)
    plt.tight_layout()
    plt.hist(array, 15, density=True)

    xx = np.linspace(np.min(array), np.max(array), 100)

    plt.plot(xx, distrs[func](xx), 'r')
    plt.title('n = %i' %i)

distrsName = ['normal', 'laplace', 'cauchy', 'poisson', 'uniform']
num = [10, 50, 1000]

for name in range(5):
    plt.figure("distribution " + distrsName[name])
    plt.title('distribution %s' %distrsName[name])

    chunk = 0#for sublot

    for i in range(3):
        draw(generateDict[distrsName[name]](num[i]), distrsName[name], chunk, num[i])
        chunk += 1

    plt.savefig(distrsName[name])