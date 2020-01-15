import matplotlib.pyplot as plt
import numpy as np
import random

c_t = [0.49999, 0.548242, 0.733333, 0.699999, 0.733333, 0.899999, 0.899999, 0.933333, 0.833333, 0.933333, 1, 1, 0.966666, 1, 0.966666, 1, 1, 1, 1]
c_t_acc = [i * 100 for i in c_t]

c_test = [0.333333, 0.433333, 0.601550, 0.700775, 0.699999, 0.711627, 0.689922, 0.713178, 0.556589, 0.689922, 0.724031, 0.727131, 0.734883, 0.714728, 0.711627, 0.722480, 0.717829, 0.691472, 0.722480]
c_test_acc = [i * 100 for i in c_test]


def linePlot():
    y1_axis = c_t_acc
    y2_axis = c_test_acc
    x_axis = np.array([x for x in range(1, len(y1_axis) + 1)])
    plt.figure()
    plt.plot(x_axis, y1_axis, '-.', linewidth=2, color='k', label='training accuracy')
    plt.plot(x_axis, y2_axis, '-.', linewidth=2, color='r', label='test accuracy')
    plt.xticks([x for x in range(0, 21, 1)])
    plt.yticks([y for y in range(50, 100, 5)])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc=2)
    ax = plt.gca()
    for x, y in zip(x_axis, y1_axis):
        ax.text(x, y, str(y), color='r', fontsize=8)
    for x, y in zip(x_axis, y2_axis):
        ax.text(x, y, str(y), color='b', fontsize=8)
    plt.show()


linePlot()
