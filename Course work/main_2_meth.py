import numpy as np
import os
import matplotlib.pyplot as plt
import decode
import sys
from main_1_meth import plot, butter_filter, init_data_

stage = 0
current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(current_dir)
try:
    import pyglobus
except ImportError as e:
    print("Cannot import pyglobus")
    sys.exit(1)

HIGH_PASS_CUTOFF = 1000
LOW_PASS_CUTOFF = 2500
MOVING_AVERAGE_WINDOW_SIZE = 1


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


if __name__ == "__main__":
    output_dir = os.path.join(current_dir, "output", "second")

    font = {"size": 22}

    plt.rc("font", **font)

    os.makedirs(output_dir, exist_ok=True)

    print("Stage %i: Data loading and preparing" % stage)

    DATA_FILE, SENSOR_NUMBER = init_data_()

    data_all = decode.extract('data', DATA_FILE, [SENSOR_NUMBER])

    c, d = decode.x_y(data_all[0][SENSOR_NUMBER])

    data = np.array([np.array(c), np.array(d)])

    print("Loaded %s" % DATA_FILE)

    roi = (300000, 310000)
    x = data[0, roi[0]:roi[1]]
    y = data[1, roi[0]:roi[1]]

    plot(data[0], data[1], "Время, с", "U, В", output_dir, color="b")

    print("Stage %i: High pass filtering" % stage)

    sample_rate = 1.0 / (x[1] - x[0])

    y = butter_filter(y, HIGH_PASS_CUTOFF, sample_rate, btype="high")

    plot(x, y, "Время, с", "U, В", output_dir, color="b")

    print("Stage %i: Low pass filtering" % stage)

    y = butter_filter(y, LOW_PASS_CUTOFF, sample_rate, btype="low")

    plot(x, y, "Время, с", "U, В", output_dir, color="b")

    print("Stage %i: Finding zero crossings" % stage)

    zero_crossings = np.where(np.diff(np.sign(y)))[0]

    plot(x, y, "Время, с", "U, В", output_dir, color="b", flush=False)
    plot(x[zero_crossings], y[zero_crossings], "Время, с", "U, В", output_dir, color="rx", new_fig=False)

    print("Stage %i: Computing frequencies" % stage)

    freqs = []

    for i in range(len(zero_crossings) - 2):
        freqs.append(1 / (x[zero_crossings[i + 2]] - x[zero_crossings[i]]))

    x = x[zero_crossings][:-(MOVING_AVERAGE_WINDOW_SIZE + 1)]
    y = moving_average(freqs, MOVING_AVERAGE_WINDOW_SIZE)

    plot(x, y, "Время, с", "Частота, Гц", output_dir, color="ko-")

    print("Done!")