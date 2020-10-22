import xlrd
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

label_path = r"E:\Research Data\Data_UF\Jule_LD14_PKYonge_Class1_Mar142019\binary_label.xlsx"
DATA_PATH = r"E:\Research Data\Data_UF\Jule_LD14_PKYonge_Class1_Mar142019\Image_Data"

GENERATE_SQUARE = 64
IMAGE_CHANNELS = 3
TRAIN_PERC = 0.6
BATCH_SIZE = 128
DROP_OUT = 0.2


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

X_with_Class_0_Num = len(list_0)
X_with_Class_1_Num = len(list_1)

ALL_DATA_NUM = X_with_Class_0_Num + X_with_Class_1_Num

class_0_data = []
for index in range(X_with_Class_0_Num):
    path = os.path.join(DATA_PATH, str(list_0[index]) + ".jpg")
    image = Image.open(path).resize((GENERATE_SQUARE, GENERATE_SQUARE), Image.ANTIALIAS)
    class_0_data.append(np.asarray(image))

print(len(class_0_data))

class_0_data = np.reshape(class_0_data, (-1, GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS))

print(len(class_0_data))

class_1_data = []
for index in range(X_with_Class_1_Num):
    path = os.path.join(DATA_PATH, str(list_1[index]) + ".jpg")
    image = Image.open(path).resize((GENERATE_SQUARE, GENERATE_SQUARE), Image.ANTIALIAS)
    class_1_data.append(np.asarray(image))

print(len(class_1_data))

class_1_data = np.reshape(class_1_data, (-1, GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS))

######split data into training and testing##################################################################

np.random.shuffle(class_0_data)
np.random.shuffle(class_1_data)

X_with_Class_0_Train_Num = int(len(class_0_data) * TRAIN_PERC)
X_with_Class_1_Train_Num = int(len(class_1_data) * TRAIN_PERC)

class_0_training_data = class_0_data[0: X_with_Class_0_Train_Num]
class_0_testing_data = class_0_data[X_with_Class_0_Train_Num:]

class_1_training_data = class_1_data[0: X_with_Class_1_Train_Num]
class_1_testing_data = class_1_data[X_with_Class_1_Train_Num:]

print("Length of class_0 test data: ", len(class_0_testing_data))
print("Length of class_1 test data: ", len(class_1_testing_data))

X_train = np.concatenate((class_0_training_data, class_1_training_data), axis=0)
y_train = np.concatenate((np.zeros((len(class_0_training_data), 1)), np.ones((len(class_1_training_data), 1))), axis=0)

X_test = np.concatenate((class_0_testing_data, class_1_testing_data), axis=0)
y_test = np.concatenate((np.zeros((len(class_0_testing_data), 1)), np.ones((len(class_1_testing_data), 1))), axis=0)
########################################################################################################

print("Start building networks...")

from numpy.random import randn
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D
from keras.layers import Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import Dropout
from sklearn.metrics import classification_report
from keras import regularizers
from tensorflow import keras

# define supervised and unsupervised discriminator models
def define_discriminator(in_shape=(GENERATE_SQUARE, GENERATE_SQUARE, 3), n_classes=2):
    # image input
    in_image = Input(shape=in_shape)
    # downsample
    fe = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(in_image)
    fe = MaxPooling2D(pool_size=(2, 2), strides=None, padding="same")(fe) # maxpooling gives almost 1.0 training accuracy
    fe = BatchNormalization(momentum=0.9)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(DROP_OUT)(fe)

    fe = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(fe)
    fe = MaxPooling2D(pool_size=(2, 2), strides=None, padding="same")(fe)
    fe = BatchNormalization(momentum=0.9)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(DROP_OUT)(fe)

    fe = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(fe)
    fe = MaxPooling2D(pool_size=(2, 2), strides=None, padding="same")(fe)
    fe = BatchNormalization(momentum=0.9)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(DROP_OUT)(fe)

    fe = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(fe)
    fe = MaxPooling2D(pool_size=(2, 2), strides=None, padding="same")(fe)
    fe = BatchNormalization(momentum=0.9)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(DROP_OUT)(fe)

    # flatten feature maps
    fe = Flatten()(fe)

    # fe = Dense(8, activation="relu", kernel_regularizer=regularizers.l2(0.2))(fe)
    # fe = Dense(4, activation="relu", kernel_regularizer=regularizers.l2(0.2))(fe)
    c_out_layer = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.2))(fe)
    c_model = Model(in_image, c_out_layer)
    c_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001, beta_1=0.5), metrics=['accuracy'])
    # sgd = keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    # c_model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return c_model


c_model = define_discriminator()
c_model.summary()

history = c_model.fit(
    X_train,
    y_train,
    batch_size = BATCH_SIZE,
    epochs = 200,
    shuffle = True,
    validation_data = (X_test, y_test)
)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()