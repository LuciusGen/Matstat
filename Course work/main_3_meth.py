import numpy as np
import os
import matplotlib.pyplot as plt
import decode
import sys
from main_1_meth import butter_filter

stage = 0
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "output", "third")

sys.path.append(current_dir)
try:
    import pyglobus
except ImportError as e:
    print("Cannot import pyglobus")
    sys.exit(1)

DATA_FILE = 0
HIGH_PASS_CUTOFF = 1000
LOW_PASS_CUTOFF = 2500
MOVING_AVERAGE_WINDOW_SIZE = 1


def plot(x, y, label_x, label_y, new_fig=True, flush=True):
    global stage

    if new_fig:
        plt.figure(figsize=(15, 10))

    global DATA_FILE

    plt.xlabel(label_x, fontsize=25)
    plt.ylabel(label_y, fontsize=25)

    plt.plot(x[0], y[0], 'ro-')
    plt.plot(x[1], y[1], 'bo-')
    plt.plot(x[2], y[2], 'ko-')
    plt.plot(x[3], y[3], 'go-')

    plt.legend(("SXR 15 мкм", "SXR 27 мкм", "SXR 50 мкм", "SXR 80 мкм"))

    if flush:
        out = os.path.join(output_dir, "pic%i.png" % stage)
        plt.savefig(out)

        print("Stage %i result:" % stage, out)

        stage += 1


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


if __name__ == "__main__":
    font = {"size": 22}
    experiments_numbers = [38993, 38994, 38995, 38996, 38998]
    roi_list = [(220000, 230000), (240000, 250000), (200000, 210000), (250000, 260000), (300000, 310000)]
    plt.rc("font", **font)

    os.makedirs(output_dir, exist_ok=True)

    print("Stage %i: Data loading and preparing" % stage)

    arr = [18, 19, 20, 26]

    for k in range(len(experiments_numbers)):
        data_array_x = []
        data_array_y = []

        for j in range(len(arr)):

            data_all = decode.extract('data', experiments_numbers[k], [arr[j]])

            c, d = decode.x_y(data_all[0][arr[j]])

            data = np.array([np.array(c), np.array(d)])

            print("Loaded %s" % DATA_FILE)

            roi = roi_list[k]
            print(roi)
            x = data[0, roi[0]:roi[1]]
            y = data[1, roi[0]:roi[1]]

            sample_rate = 1.0 / (x[1] - x[0])

            y = butter_filter(y, HIGH_PASS_CUTOFF, sample_rate, btype="high")

            y = butter_filter(y, LOW_PASS_CUTOFF, sample_rate, btype="low")

            zero_crossings = np.where(np.diff(np.sign(y)))[0]

            print("Stage %i: Computing frequencies" % stage)

            freqs = []

            for i in range(len(zero_crossings) - 2):
                freqs.append(1 / (x[zero_crossings[i + 2]] - x[zero_crossings[i]]))

            x = x[zero_crossings][:-(MOVING_AVERAGE_WINDOW_SIZE + 1)]
            y = moving_average(freqs, MOVING_AVERAGE_WINDOW_SIZE)

            data_array_x.append(x)
            data_array_y.append(y)

        plot(data_array_x, data_array_y, "Время, с", "Частота, Гц")