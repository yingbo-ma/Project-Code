import xlrd
import os
from PIL import Image
import numpy as np

train_label_path = r"D:\Data\Data_UF\Yingbo_LD2_PKYonge_Class1_Mar142019_B\binary_label.xlsx"
train_DATA_PATH = r"D:\Data\Data_UF\Yingbo_LD2_PKYonge_Class1_Mar142019_B\Image_Data"

test_label_path = r"D:\Data\Data_UF\Jule_LD14_PKYonge_Class1_Mar142019\binary_label.xlsx"
test_DATA_PATH = r"D:\Data\Data_UF\Jule_LD14_PKYonge_Class1_Mar142019\Image_Data"

GENERATE_SQUARE = 64
IMAGE_CHANNELS = 3
TRAIN_PERC = 0.75
CLASS_NUM = 2
BATCH_SIZE = 100
n_samples = int(BATCH_SIZE / CLASS_NUM)
latent_dim = 100
IMAGE_NUM = 2574
BATCH_NUM = int(IMAGE_NUM / BATCH_SIZE) + 1
ITERATIONS = 3000


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


print("Start reading training Image & Label data...")

train_list = excel_data(train_label_path)

train_list_0 = []
for i, j in enumerate(train_list):
    if j == 0:
        train_list_0.append(i)

train_list_1 = []
for i, j in enumerate(train_list):
    if j == 1:
        train_list_1.append(i)

train_class_0_data = []
for index in range(len(train_list_0)):
    path = os.path.join(train_DATA_PATH, str(train_list_0[index]) + ".jpg")
    image = Image.open(path).resize((GENERATE_SQUARE, GENERATE_SQUARE), Image.ANTIALIAS)
    train_class_0_data.append(np.asarray(image))

print(len(train_class_0_data))

train_class_0_data = np.reshape(train_class_0_data, (-1, GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS))
train_class_0_data = train_class_0_data[0 : 900]
print(len(train_class_0_data))

# we can cut some 0 data here to improve training performance

train_class_1_data = []
for index in range(len(train_list_1)):
    path = os.path.join(train_DATA_PATH, str(train_list_1[index]) + ".jpg")
    image = Image.open(path).resize((GENERATE_SQUARE, GENERATE_SQUARE), Image.ANTIALIAS)
    train_class_1_data.append(np.asarray(image))

print(len(train_class_1_data))

train_class_1_data = np.reshape(train_class_1_data, (-1, GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS))

train_data = np.concatenate((train_class_0_data, train_class_1_data), axis=0)

print(train_data.shape)

########################################################################################################
print("Start reading testing Image & Label data...")

test_list = excel_data(test_label_path)

test_list_0 = []
for i, j in enumerate(test_list):
    if j == 0:
        test_list_0.append(i)

test_list_1 = []
for i, j in enumerate(test_list):
    if j == 1:
        test_list_1.append(i)

test_class_0_data = []
for index in range(len(test_list_0)):
    path = os.path.join(test_DATA_PATH, str(test_list_0[index]) + ".jpg")
    image = Image.open(path).resize((GENERATE_SQUARE, GENERATE_SQUARE), Image.ANTIALIAS)
    test_class_0_data.append(np.asarray(image))

print(len(test_class_0_data))

test_class_0_data = np.reshape(test_class_0_data, (-1, GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS))

# we can cut some 0 data here to improve training performance

test_class_1_data = []
for index in range(len(test_list_1)):
    path = os.path.join(test_DATA_PATH, str(test_list_1[index]) + ".jpg")
    image = Image.open(path).resize((GENERATE_SQUARE, GENERATE_SQUARE), Image.ANTIALIAS)
    test_class_1_data.append(np.asarray(image))

print(len(test_class_1_data))

test_class_1_data = np.reshape(test_class_1_data, (-1, GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS))

test_data = np.concatenate((test_class_0_data, test_class_1_data), axis=0)
y_test = np.concatenate((np.zeros((len(test_class_0_data), 1)), np.ones((len(test_class_1_data), 1))), axis=0)

print(test_data.shape)


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
    fe = Conv2D(32, (3, 3), strides=(2, 2), padding='same')(in_image)
    fe = BatchNormalization(momentum=0.9)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.2)(fe)

    fe = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(fe)
    fe = BatchNormalization(momentum=0.9)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.2)(fe)

    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe)
    fe = BatchNormalization(momentum=0.9)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.2)(fe)

    fe = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(fe)
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
c_model.summary()

print("Start training...")

epoch = 0

for i in range(ITERATIONS):
    ####generate supervised real data
    ix = np.random.randint(0, len(train_class_0_data), n_samples)
    X_supervised_samples_class_0 = np.asarray(train_class_0_data[ix])
    Y_supervised_samples_class_0 = np.zeros((n_samples, 1))

    ix = np.random.randint(0, len(train_class_1_data), n_samples)
    X_supervised_samples_class_1 = np.asarray(train_class_1_data[ix])
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
        _, test_acc = c_model.evaluate(test_data, y_test, verbose=0)
        print(f"Epoch {epoch}, c model accuracy on test data: {test_acc}")
        y_pred = c_model.predict(test_data, batch_size=60, verbose=0)

        pred_list = y_pred.tolist()

        for i in range(len(pred_list)):
            if pred_list[i] > [0.5]:
                pred_list[i] = [1]
            else:
                pred_list[i] = [0]

        y_pred = np.asarray(pred_list)
        print("Length of y_pred: ", len(y_pred))
        print(classification_report(y_test, y_pred))
