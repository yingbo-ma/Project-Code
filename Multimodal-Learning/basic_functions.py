import xlrd
import os
from PIL import Image
import numpy as np

from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras import regularizers

def read_excel(label_file_path):
    data = xlrd.open_workbook(label_file_path)
    table = data.sheet_by_index(0)

    nrows = table.nrows
    ncols = table.ncols

    excel_list = []
    for row in range(0, nrows):
        for col in range(ncols):
            cell_value = int(table.cell(row, col).value)
            excel_list.append(cell_value)
    return excel_list

def data_prepare(excel_list, image_data_path, pixel, num_channels):

    class_0_list = []
    class_1_list = []
    class_2_list = []

    class_0_data = []
    class_1_data = []
    class_2_data = []

    for i, j in enumerate(excel_list):
        if j == 0:
            class_0_list.append(i)

    for index in range(len(class_0_list)):
        path = os.path.join(image_data_path, str(class_0_list[index]) + ".jpg")
        image = Image.open(path).resize((pixel, pixel), Image.ANTIALIAS)
        class_0_data.append(np.asarray(image))

    class_0_data = np.reshape(class_0_data, (-1, pixel, pixel, num_channels))

    for i, j in enumerate(excel_list):
        if j == 1:
            class_1_list.append(i)

    for index in range(len(class_1_list)):
        path = os.path.join(image_data_path, str(class_1_list[index]) + ".jpg")
        image = Image.open(path).resize((pixel, pixel), Image.ANTIALIAS)
        class_1_data.append(np.asarray(image))

    class_1_data = np.reshape(class_1_data, (-1, pixel, pixel, num_channels))

    for i, j in enumerate(excel_list):
        if j == 2:
            class_2_list.append(i)

    for index in range(len(class_2_list)):
        path = os.path.join(image_data_path, str(class_2_list[index]) + ".jpg")
        image = Image.open(path).resize((pixel, pixel), Image.ANTIALIAS)
        class_2_data.append(np.asarray(image))

    class_2_data = np.reshape(class_2_data, (-1, pixel, pixel, num_channels))

    print("data_0.shape: ", class_0_data.shape)
    print("data_1.shape: ", class_1_data.shape)
    print("data_2.shape: ", class_2_data.shape)

    return class_0_data, class_1_data, class_2_data

def train_test_data_split(data_0, data_1, data_2, split_ratio):
    np.random.shuffle(data_0)
    np.random.shuffle(data_1)
    np.random.shuffle(data_2)

    Class_0_Train_Num = int(len(data_0) * split_ratio)
    Class_1_Train_Num = int(len(data_1) * split_ratio)
    Class_2_Train_Num = int(len(data_2) * split_ratio)

    class_0_train_data = data_0[0: Class_0_Train_Num]
    class_0_test_data = data_0[Class_0_Train_Num : ]

    class_1_train_data = data_1[0: Class_1_Train_Num]
    class_1_test_data = data_1[Class_1_Train_Num : ]

    class_2_train_data = data_2[0: Class_2_Train_Num]
    class_2_test_data = data_2[Class_2_Train_Num : ]

    train_data = np.concatenate((class_0_train_data, class_1_train_data, class_2_train_data), axis=0)
    test_data = np.concatenate((class_0_test_data, class_1_test_data, class_2_test_data), axis=0)
    test_label = np.concatenate((np.zeros((len(class_0_test_data), 1)), np.ones((len(class_1_test_data), 1)), 2 * np.ones((len(class_2_test_data), 1))), axis=0)

    print("data_train.shape: ", train_data.shape)
    print("data_test.shape: ", test_data.shape)
    print("label_test.shape: ", test_label.shape)

    return class_0_train_data, class_1_train_data, class_2_train_data, train_data, test_data, test_label

def define_model(input_shape, n_classes):

    in_image = Input(shape=input_shape)

    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(in_image)
    fe = BatchNormalization(momentum=0.9)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.15)(fe)

    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe)
    fe = BatchNormalization(momentum=0.9)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.15)(fe)

    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe)
    fe = BatchNormalization(momentum=0.9)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.15)(fe)

    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe)
    fe = BatchNormalization(momentum=0.9)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.15)(fe)

    fe = Flatten()(fe)

    c_out_layer = Dense(n_classes, activation="softmax", kernel_regularizer=regularizers.l2(0.01))(fe)
    c_model = Model(in_image, c_out_layer)
    c_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])

    c_model.summary()

    return c_model
