import numpy
import matplotlib.pyplot as plt
import sys


POISSON_PARAM = 10
UNIFORM_LEFT = -numpy.sqrt(3)
UNIFORM_RIGHT = numpy.sqrt(3)
LAPLAS_COEF = numpy.sqrt(2)
selection = [20, 100]
selection = numpy.sort(selection)


def standard_normal(x):
    return (1 / numpy.sqrt(2*numpy.pi)) * numpy.exp(- x * x / 2)


def standard_cauchy(x):
    return 1 / (numpy.pi * (1 + x*x))


def laplace(x):
    return 1 / LAPLAS_COEF * numpy.exp(-LAPLAS_COEF * numpy.abs(x))


def uniform(x):
    flag2 = x <= UNIFORM_RIGHT
    flag1 = x >= UNIFORM_LEFT
    return 1 / (UNIFORM_RIGHT - UNIFORM_LEFT) * flag1 * flag2


def poisson(x):
    k = POISSON_PARAM
    return (numpy.power(x, k) / numpy.math.factorial(k)) * numpy.exp(-x)


func_dict = {
    'normal': standard_normal,
    'cauchy': standard_cauchy,
    'laplace': laplace,
    'uniform': uniform,
    'poisson': poisson
}


def generate_laplace(x):
    return numpy.random.laplace(0, 1/LAPLAS_COEF, x)


def generate_uniform(x):
    return numpy.random.uniform(UNIFORM_LEFT, UNIFORM_RIGHT, x)


def generate_poisson(x):
    return numpy.random.poisson(POISSON_PARAM, x)


generate_dict = {
    'normal': numpy.random.standard_normal,
    'cauchy': numpy.random.standard_cauchy,
    'laplace': generate_laplace,
    'uniform': generate_uniform,
    'poisson': generate_poisson
}


def z_r(x):
    return (numpy.amin(x) + numpy.amax(x))/2


def z_q(x):
    return (numpy.quantile(x, 1/4) + numpy.quantile(x, 3/4)) / 2


def z_tr(x):
    length = x.size
    r = int(length / 4)
    counter = 0

    for i in range(r, length - r):
        counter += x[i]
    return counter/(length - 2 * r)


def i_q_r(x):
    return numpy.abs(numpy.quantile(x, 1 / 4) - numpy.quantile(x, 3 / 4))


def ejection(x):
    length = x.size
    count = 0
    top = numpy.quantile(x, 1 / 4) - 1.5 * i_q_r(x)
    bottom = numpy.quantile(x, 3 / 4) + 1.5 * i_q_r(x)

    for i in range(0, length):
        if x[i] < top or x[i] > bottom:
            count += 1
    return count / length


pos_characteristic_dict = {
    'average': numpy.mean,
    'med': numpy.median,
    'Zr': z_r,
    'Zq': z_q,
    'Ztr r = n/4': z_tr
}

pos_char_name = [
    'average',
    'med',
    'Zr',
    'Zq',
    'Ztr r = n/4'
]


def e(z):
    return numpy.mean(z)


f = open('out1.txt', 'w')
std = sys.stdout
sys.stdout = f


def research(dist_type):
    print()
    print(dist_type)

    data = []

    for num in selection:
        eject = []
        arr = numpy.sort(generate_dict[dist_type](num))
        data.append(arr)

        for i in range(0, 1000):
            arr = numpy.sort(generate_dict[dist_type](num))
            eject.append(ejection(arr))

        print("%-10s;" % ('n = %i' % num), end="")
        print("%-12f;" % e(eject), end="")
        print()

    plt.figure(dist_type)
    plt.title(dist_type)
    plt.subplots()
    plt.xlabel("x")
    plt.ylabel("n")

    plt.boxplot(data, vert=False)

    plt.savefig(dist_type)


research('normal')
research('cauchy')
research('laplace')
research('uniform')
research('poisson')


f.close()
sys.stdout = std
