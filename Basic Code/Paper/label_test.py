import xlrd
import os
from PIL import Image
import numpy as np

label_path = r"D:\Data\LD2_PKYonge_Class1_Mar142019_B\binary_label.xlsx"
DATA_PATH = r"D:\Data\LD2_PKYonge_Class1_Mar142019_B\Image_Data"


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


print("Start reading Image & Label data...")

label_list = excel_data(label_path)
target_index = [[[i + j] for i in range(5)] for j in range(2574 - 5 + 1)]
target = [[]]

for i in target_index:
    temp = []
    temp.append(label_list[i[0][0]])
    temp.append(label_list[i[1][0]])
    temp.append(label_list[i[2][0]])
    temp.append(label_list[i[3][0]])
    temp.append(label_list[i[4][0]])
    target.append(temp)

target.remove([])
target = np.reshape(target, (-1, 5, 1))
print(target.shape)

original_data = []

for j in range(len(label_list)):
    path = os.path.join(DATA_PATH, str(j) + ".jpg")
    image = Image.open(path).resize((64, 64), Image.ANTIALIAS)
    original_data.append(np.asarray(image))

original_data = np.reshape(original_data, (-1, 64, 64, 3))

data_index = [[[i + j] for i in range(5)] for j in range(2574 - 5 + 1)]
data = [[]]

for k in data_index:
    temp = []
    temp.append(original_data[k[0][0]])
    temp.append(original_data[k[1][0]])
    temp.append(original_data[k[2][0]])
    temp.append(original_data[k[3][0]])
    temp.append(original_data[k[4][0]])
    data.append(temp)

data.remove([])
data = np.reshape(data, (-1, 5, 64, 64, 3))
print(data.shape)





