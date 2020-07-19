import matplotlib.pyplot as plt
import numpy as np
import random

precison = [0.46, 0.64, 0.87, 0.67, 0.66, 0.61, 0.73, 0.67, 0.71, 0.69, 0.67, 0.68, 0.68, 0.68, 0.68, 0.72, 0.76, 0.68]
precision_acc = [i * 100 for i in precison]

recall = [0.95, 0.85, 0.44, 0.73, 0.78, 0.84, 0.66, 0.76, 0.73, 0.72, 0.76, 0.74, 0.76, 0.76, 0.76, 0.69, 0.76, 0.74]
recall_acc = [i * 100 for i in recall]

f1 = [0.62, 0.73, 0.58, 0.70, 0.71, 0.71, 0.69, 0.71, 0.72, 0.70, 0.71, 0.71, 0.72, 0.72, 0.72, 0.70, 0.76, 0.71]
f1_acc = [i * 100 for i in f1]


def linePlot():
    y1_axis = precision_acc
    y2_axis = recall_acc
    y3_axis = f1_acc
    x_axis = np.array([x for x in range(1, len(y1_axis) + 1)])
    plt.figure()
    plt.plot(x_axis, y1_axis, '-.', linewidth=1.5, color='c', label='precision for speaking')
    plt.plot(x_axis, y2_axis, '-.', linewidth=1.5, color='r', label='recall for speaking')
    plt.plot(x_axis, y3_axis, '-', linewidth=3.5, color='k', label='f1-score for speaking')
    plt.xticks([x for x in range(0, 21, 1)])
    plt.yticks([y for y in range(50, 100, 5)])
    plt.xlabel("Epoch")
    plt.legend(loc=4)
    plt.title("CNN+LSTM for Speech Detection")
    ax = plt.gca()
    for x, y in zip(x_axis, y1_axis):
        ax.text(x, y, str(y), color='c', fontsize=8)
    for x, y in zip(x_axis, y2_axis):
        ax.text(x, y, str(y), color='r', fontsize=8)
    plt.show()

linePlot()
