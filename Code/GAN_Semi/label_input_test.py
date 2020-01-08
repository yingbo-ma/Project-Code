import xlrd
import os
from PIL import Image
import numpy as np

label_path = r"D:\Data\new_label.xlsx"

GENERATE_SQUARE = 64
IMAGE_CHANNELS = 3
TRAIN_PERC = 0.75
BATCH_SIZE = 30
CLASS_NUM = 3
n_samples = int(BATCH_SIZE / CLASS_NUM)
latent_dim = 100
EPOCHS = 10000

### get the all data for 3 classes ######################################################################################################
def excel_data(file_path):
    data = xlrd.open_workbook(file_path)
    table = data.sheet_by_index(0)

    nrows = table.nrows
    ncols = table.ncols
    
    excel_list = []
    for row in range(0, nrows):
        for col in range(ncols):
            cell_value = int(table.cell(row, col).value)
            excel_list.append(cell_value)
    return excel_list

list = excel_data(label_path)

list_0 = []
for i, j in enumerate(list):
    if j == 0:
        list_0.append(i)

list_1 = []
for i, j in enumerate(list):
    if j == 1:
        list_1.append(i)

list_2 = []
for i, j in enumerate(list):
    if j == 2:
        list_2.append(i)

list_3 = []
for i, j in enumerate(list):
    if j == 3:
        list_3.append(i)

print(len(list_0))
print(len(list_1))
print(len(list_2))
print(len(list_3))
print(len(list_0)+len(list_1)+len(list_2)+len(list_3))