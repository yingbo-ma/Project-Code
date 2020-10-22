import numpy as np

class_0_labels = np.zeros((25, 1))
class_1_labels = np.ones((25, 1))
class_2_labels = 2 * np.ones((25, 1))
class_3_labels = 3 * np.ones((25, 1))

Labels = np.concatenate((class_0_labels, class_1_labels, class_2_labels, class_3_labels), axis=0)
print(Labels.shape)




