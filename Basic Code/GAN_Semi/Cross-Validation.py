import xlrd
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import KFold

label_path = r"D:\Data\LD2_PKYonge_Class1_Mar142019_B\new_label.xlsx"
DATA_PATH = r"D:\Data\LD2_PKYonge_Class1_Mar142019_B\Image_Data"

GENERATE_SQUARE = 64
IMAGE_CHANNELS = 3
TRAIN_PERC = 0.75
CLASS_NUM = 3
BATCH_SIZE = 30
n_samples = int(BATCH_SIZE / CLASS_NUM)
latent_dim = 100
IMAGE_NUM = 2574
BATCH_NUM = int(IMAGE_NUM / BATCH_SIZE) + 1
ITERATIONS = 2000


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

X_with_Class_0_Num = len(list_0)
X_with_Class_1_Num = len(list_1)
X_with_Class_2_Num = len(list_2)

ALL_DATA_NUM = X_with_Class_0_Num + X_with_Class_1_Num + X_with_Class_2_Num

X_with_Class_0_Train_Num = int(X_with_Class_0_Num * TRAIN_PERC)
X_with_Class_1_Train_Num = int(X_with_Class_1_Num * TRAIN_PERC)
X_with_Class_2_Train_Num = int(X_with_Class_2_Num * TRAIN_PERC)

class_0_data = []
for index in range(X_with_Class_0_Num):
    path = os.path.join(DATA_PATH, str(list_0[index]) + ".jpg")
    image = Image.open(path).resize((GENERATE_SQUARE, GENERATE_SQUARE), Image.ANTIALIAS)
    class_0_data.append(np.asarray(image))

class_0_data = np.reshape(class_0_data, (-1, GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS))

class_1_data = []
for index in range(X_with_Class_1_Num):
    path = os.path.join(DATA_PATH, str(list_1[index]) + ".jpg")
    image = Image.open(path).resize((GENERATE_SQUARE, GENERATE_SQUARE), Image.ANTIALIAS)
    class_1_data.append(np.asarray(image))

class_1_data = np.reshape(class_1_data, (-1, GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS))

class_2_data = []
for index in range(X_with_Class_2_Num):
    path = os.path.join(DATA_PATH, str(list_2[index]) + ".jpg")
    image = Image.open(path).resize((GENERATE_SQUARE, GENERATE_SQUARE), Image.ANTIALIAS)
    class_2_data.append(np.asarray(image))

class_2_data = np.reshape(class_2_data, (-1, GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS))

######randomly shuffle the dataset##################################################################

np.random.shuffle(class_0_data)
np.random.shuffle(class_1_data)
np.random.shuffle(class_2_data)

#######prepare the dataset for K-fold validation###########################################################################
np.random.shuffle(class_0_data)
np.random.shuffle(class_1_data)
np.random.shuffle(class_2_data)

class_0_training_data = class_0_data[0: X_with_Class_0_Train_Num]
class_0_testing_data = class_0_data[X_with_Class_0_Train_Num:]

class_1_training_data = class_1_data[0: X_with_Class_1_Train_Num]
class_1_testing_data = class_1_data[X_with_Class_1_Train_Num:]

class_2_training_data = class_2_data[0: X_with_Class_2_Train_Num]
class_2_testing_data = class_2_data[X_with_Class_2_Train_Num:]

print("Length of class_0 test data: ", len(class_0_testing_data))
print("Length of class_1 test data: ", len(class_1_testing_data))
print("Length of class_2 test data: ", len(class_2_testing_data))

X_test = np.concatenate((class_0_testing_data, class_1_testing_data, class_2_testing_data), axis=0)
y_test = np.concatenate((np.zeros((len(class_0_testing_data), 1)), np.ones((len(class_1_testing_data), 1)),
                         2 * np.ones((len(class_2_testing_data), 1))), axis=0)

print(y_test)

