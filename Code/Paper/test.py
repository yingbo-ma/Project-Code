import xlrd
import os
from PIL import Image
import numpy as np


a = ((1, 2, 3))
b = ((1, 2, 3))
c = ((1, 2, 3))

y = np.concatenate((a, b, c), axis=0)

print(y)