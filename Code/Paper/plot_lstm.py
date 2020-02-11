import matplotlib.pyplot as plt
import numpy as np
import random

precison = [0.42, 0.41, 0.43, 0.42, 0.43, 0.45, 0.44, 0.45, 0.45, 0.45, 0.44, 0.42, 0.43, 0.42, 0.42, 0.43, 0.42, 0.42]
precision_acc = [i * 100 for i in precison]

recall = [0.52, 0.51, 0.53, 0.51, 0.52, 0.57, 0.55, 0.57, 0.55, 0.55, 0.54, 0.53, 0.53, 0.52, 0.50, 0.55, 0.53, 0.52]
recall_acc = [i * 100 for i in recall]

f1 = [0.47, 0.46, 0.47, 0.46, 0.47, 0.50, 0.49, 0.51, 0.49, 0.50, 0.48, 0.47, 0.48, 0.47, 0.46, 0.48, 0.46, 0.46]
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
    plt.yticks([y for y in range(30, 80,  5)])
    plt.xlabel("Epoch")
    plt.legend(loc=2)
    plt.title("LSTM for Speech Detection")
    ax = plt.gca()
    for x, y in zip(x_axis, y1_axis):
        ax.text(x, y, str(y), color='c', fontsize=8)
    for x, y in zip(x_axis, y2_axis):
        ax.text(x, y, str(y), color='r', fontsize=8)
    plt.show()

linePlot()
