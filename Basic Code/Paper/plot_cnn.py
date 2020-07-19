import matplotlib.pyplot as plt
import numpy as np
import random

precison = [0.60, 0.69, 0.75, 0.70, 0.71, 0.83, 0.80, 0.77, 0.74, 0.74, 0.77, 0.81, 0.65, 0.81, 0.78, 0.85, 0.69, 0.67]
precision_acc = [i * 100 for i in precison]

recall = [0.77, 0.66, 0.64, 0.74, 0.74, 0.52, 0.65, 0.74, 0.75, 0.72, 0.68, 0.58, 0.82, 0.54, 0.69, 0.46, 0.78, 0.81]
recall_acc = [i * 100 for i in recall]

f1 = [0.67, 0.68, 0.69, 0.72, 0.73, 0.64, 0.72, 0.75, 0.75, 0.73, 0.72, 0.68, 0.72, 0.64, 0.73, 0.60, 0.73, 0.73]
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
    plt.legend(loc=2)
    plt.title("CNN for Speech Detection")
    ax = plt.gca()
    for x, y in zip(x_axis, y1_axis):
        ax.text(x, y, str(y), color='c', fontsize=8)
    for x, y in zip(x_axis, y2_axis):
        ax.text(x, y, str(y), color='r', fontsize=8)
    plt.show()

linePlot()
