import xlrd
import os
from PIL import Image
import numpy as np

label_path = r"D:\Data\LD2_PKYonge_Class1_Mar142019_B\binary_label.xlsx"
DATA_PATH = r"D:\Data\LD2_PKYonge_Class1_Mar142019_B\Image_Data"

GENERATE_SQUARE = 64
IMAGE_CHANNELS = 3
TRAIN_PERC = 0.75
CLASS_NUM = 2
BATCH_SIZE = 60
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

class_0_data = class_0_data[0: 900]
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
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import Dropout
from sklearn.metrics import classification_report
from keras import regularizers


# define supervised and unsupervised discriminator models
def define_discriminator(in_shape=(64, 64, 3), n_classes=2):
    # image input
    in_image = Input(shape=in_shape)
    # downsample
    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(in_image)
    fe = BatchNormalization(momentum=0.9)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.2)(fe)

    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe)
    fe = BatchNormalization(momentum=0.9)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.2)(fe)

    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe)
    fe = BatchNormalization(momentum=0.9)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.2)(fe)

    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe)
    fe = BatchNormalization(momentum=0.9)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.2)(fe)

    # flatten feature maps
    fe = Flatten()(fe)

    c_out_layer = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01))(fe)
    c_model = Model(in_image, c_out_layer)
    c_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])

    return c_model


c_model = define_discriminator()

print("Start training...")

epoch = 0

for i in range(ITERATIONS):
    ####generate supervised real data
    ix = np.random.randint(0, len(class_0_training_data), n_samples)
    X_supervised_samples_class_0 = np.asarray(class_0_training_data[ix])
    Y_supervised_samples_class_0 = np.zeros((n_samples, 1))

    ix = np.random.randint(0, len(class_1_training_data), n_samples)
    X_supervised_samples_class_1 = np.asarray(class_1_training_data[ix])
    Y_supervised_samples_class_1 = np.ones((n_samples, 1))

    Xsup_real = np.concatenate(
        (X_supervised_samples_class_0, X_supervised_samples_class_1), axis=0)
    ysup_real = np.concatenate(
        (Y_supervised_samples_class_0, Y_supervised_samples_class_1), axis=0)

    # update supervised discriminator (c)
    c_loss, c_acc = c_model.train_on_batch(Xsup_real, ysup_real)

    if (i + 1) % (BATCH_NUM * 1) == 0:
        epoch += 1
        print(f"Epoch {epoch}, c model accuracy on training data: {c_acc}")
        _, test_acc = c_model.evaluate(X_test, y_test, verbose=0)
        print(f"Epoch {epoch}, c model accuracy on test data: {test_acc}")
        y_pred = c_model.predict(X_test, batch_size=60, verbose=0)

        pred_list = y_pred.tolist()

        for i in range(len(pred_list)):
            if pred_list[i] > [0.5]:
                pred_list[i] = [1]
            else:
                pred_list[i] = [0]

        y_pred = np.asarray(pred_list)
        print("Length of y_pred: ", len(y_pred))
        print(classification_report(y_test, y_pred))