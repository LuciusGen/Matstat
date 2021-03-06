import datetime
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import decode
import os
import sys


stage = 0
current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(current_dir)
try:
    import pyglobus
except ImportError as e:
    print("Cannot import pyglobus")
    sys.exit(1)

HIGH_PASS_CUTOFF = 850
SMOOTHED_DD1_ORDER = 50
LOW_PASS_CUTOFF = 5 * HIGH_PASS_CUTOFF
SAWTOOTH_DETECTION_THRESHOLD = 0.00004


def plot(x, y, label_x, label_y, output_dir, color="k", new_fig=True, flush=True):
    global stage

    if new_fig:
        plt.figure(figsize=(15, 10))

    plt.plot(x, y, color)
    plt.xlabel(label_x, fontsize=25)
    plt.ylabel(label_y, fontsize=25)

    if flush:
        out = os.path.join(output_dir, "pic%i.png" % stage)
        plt.savefig(out)
        plt.close()

        print("Stage %i result:" % stage, out)

        stage += 1


def butter_filter(input_, cutoff, fs, btype, order=5):
    b, a = signal.butter(order, cutoff / (0.5 * fs), btype=btype, analog=False)
    return signal.filtfilt(b, a, input_)


# Applies threshold to processed data and return relative (in ROI domain) indexes of sawtooth start and end
def get_sawtooth_indexes(y, threshold):
    start_ind = 0
    end_index = 0

    data_length = y.shape[0]

    for i in range(data_length):
        if y[i] >= threshold:
            start_ind = i
            break

    for i in range(1, data_length):
        if y[data_length - i] >= threshold:
            end_index = data_length - i
            break

    return start_ind, end_index


# Computing smoothed first derivative
def smoothed_dd1(input_, order):
    input_ = input_.astype(np.float32)
    coeff = 1.0 / order / (order + 1)

    data_size = np.shape(input_)[0]

    output = np.zeros(data_size)

    for i in range(data_size):
        for c in range(1, order + 1):
            output[i] += coeff * (input_[min(i + c, data_size - 1)] - input_[max(i - c, 0)])

    return output


# Enter the needed values to start algoritm
def init_data_():
    experiments_numbers = [38993, 38994, 38995, 38996, 38998]
    name_detectors = [15, 27, 50, 80]
    arr = [18, 19, 20, 26]

    print("Enter the number of experiment: \n")

    for i in range(len(experiments_numbers)):
        print(experiments_numbers[i])

    print("\n")

    DATA_FILE = int(input())

    c = False
    for i in range(len(experiments_numbers)):
        if DATA_FILE == experiments_numbers[i]:
            c = True
            break

    if c != True:
        print("Bad input!")
        exit(1)

    os.system('cls')

    print("Enter the number of SXR Detector: ")

    for i in range(len(name_detectors)):
        print("SXR ", name_detectors[i], " mkm")

    print("\n")

    SENSOR_NUMBER = int(input())

    os.system('cls')

    d = False
    for i in range(len(name_detectors)):
        if SENSOR_NUMBER == name_detectors[i]:
            d = True
            SENSOR_NUMBER = arr[i]
            break

    if d != True:
        print("Bad input!")
        exit(1)

    os.system('cls')

    return DATA_FILE, SENSOR_NUMBER

if __name__ == "__main__":
    output_dir = os.path.join(current_dir, "output", "first")

    start = datetime.datetime.now()
    font = {"size": 22}

    plt.rc("font", **font)

    os.makedirs(output_dir, exist_ok=True)

    print("Stage %i: Data loading" % stage)

    DATA_FILE, SENSOR_NUMBER = init_data_()

    data_all = decode.extract('data', DATA_FILE, [SENSOR_NUMBER])

    c, d = decode.x_y(data_all[0][SENSOR_NUMBER])

    data = np.array([np.array(c), np.array(d)])

    print("Loaded %s" % DATA_FILE)

    plot(data[0], data[1], "Время, с", "U, В", output_dir, color='b')

    print("Stage %i: ROI extracting" % stage)

    roi = pyglobus.sawtooth.get_signal_roi(data[1], mean_scale=1)
    print(roi)
    x = data[0, roi[0]:roi[1]]
    y = data[1, roi[0]:roi[1]]

    plot(x, y, "Время, с", "U, В", output_dir, color='b')

    print("Stage %i: High pass filtering" % stage)

    sample_rate = 1.0 / (x[1] - x[0])

    y = butter_filter(y, HIGH_PASS_CUTOFF, sample_rate, btype="highpass")

    plot(x, y, "Время, с", "U, В", output_dir, color='b')

    print("Stage %i: Smoothed differentiation" % stage)

    y = smoothed_dd1(y, SMOOTHED_DD1_ORDER)

    plot(x, y, "Время, с", "U', В/с", output_dir, color='b')

    print("Stage %i: Module + Low pass filtering" % stage)

    y = np.abs(y)
    y = butter_filter(y, LOW_PASS_CUTOFF, sample_rate, btype="low")

    plot(x, y, "Время, с", "|U'|, В/с", output_dir, color='b')

    print("Stage %i: Sawtooth detection" % stage)

    plot(x, y, "Время, с", "|U'|, В/с", output_dir, flush=False, color='b')
    start_ind1, end_ind1 = get_sawtooth_indexes(y, SAWTOOTH_DETECTION_THRESHOLD)
    plt.axvline(x[start_ind1], color="k")
    plt.axvline(x[end_ind1], color="k")
    plot(x, [SAWTOOTH_DETECTION_THRESHOLD] * len(x), "Время, с", "|U'|, В/с", output_dir, color="r", new_fig=False)

    print("Done!", "time = ", str(datetime.datetime.now() - start))