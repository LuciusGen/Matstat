import numpy as np
import matplotlib.pyplot as plt
import sys

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


def Zr(x):
    return (np.amin(x) + np.amax(x))/2


def Zq(x):
    return (np.quantile(x, 1/4) + np.quantile(x, 3/4) ) /2


def Ztr(x):
    length = x.size
    r = (int)(length / 4)
    sum = 0
    for i in range(r, length - r):
        sum += x[i]
    return sum/(length - 2 * r)


posCharacteristicDict = {
    'average': np.mean,
    'med': np.median,
    'Zr': Zr,
    'Zq': Zq,
    'Ztr r = n/4': Ztr
}

posNames = [
    'average',
    'med',
    'Zr',
    'Zq',
    'Ztr r = n/4'
]


def E(z):
    return np.mean(z)


def D(z):
    return np.var(z)


numList = [10, 50, 1000]
sys.stdout = open('out.txt', 'w')


def research(distType):
    print('-------------------------------------')
    print(distType)
    for num in numList:
        printTable = {
            'E': [],
            'D': []
        }
        for name in posNames:
            z = []
            for i in range(0, 1000):
                arr = np.sort(generateDict[distType](num))
                z.append(posCharacteristicDict[name](arr))
            printTable['E'].append(E(z))
            printTable['D'].append(D(z))

        printList(num, posNames)
        printED('E =', printTable['E'])
        printED('D =', printTable['D'])

        print()


def printList(num, listi):
    print()
    print("%-9s" % ('n = %i' % num), end="")
    for i in listi:
        print("%-17s" % i, end="")


def printED(ED, printTableED):
    print()
    print("%-9s" % (ED), end="")
    for e in printTableED:
        print("%-17f" % e, end="")


research('normal')
research('cauchy')
research('laplace')
research('uniform')
research('poisson')