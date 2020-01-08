import matplotlib.pyplot as plt
import numpy as np
import random

c_ = [0.633333, 0.733333, 0.866666, 0.866666, 0.933333, 0.966666, 0.966666, 0.833333, 0.733333, 1, 1, 1, 0.966666, 1, 1, 1, 1, 1, 0.766666, 0.966666, 1, 1, 1, 1, 1, 1, 1, 1, 1]
c_acc = [i * 100 for i in c_]

def linePlot():
    y_axis = c_acc
    x_axis = np.array([x for x in range(1, len(y_axis)+1)])
    plt.figure()
    plt.plot(x_axis, y_axis, '-.', linewidth = 2, color = 'k', label = 'accuracy')
    plt.yticks([y for y in range(50, 100, 5)])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc = 2)
    ax = plt.gca()
    for x, y in zip(x_axis, y_axis):
        ax.text(x, y, str(y), color = 'r', fontsize = 8)
    plt.show()

linePlot()