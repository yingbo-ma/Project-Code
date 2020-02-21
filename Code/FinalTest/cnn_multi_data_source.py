import xlrd
import os
from PIL import Image
import numpy as np

test_label_path_0 = r"D:\Data\Data_NC_State\TU409-10B\binary_label.xlsx"
test_DATA_PATH_0 = r"D:\Data\Data_NC_State\TU409-10B\Image_Data"

test_label_path_1 = r"D:\Data\Data_UF\Jule_LD14_PKYonge_Class1_Mar142019\binary_label.xlsx"
test_DATA_PATH_1 = r"D:\Data\Data_UF\Jule_LD14_PKYonge_Class1_Mar142019\Image_Data"

train_label_path_0 = r"D:\Data\Data_NC_State\TU405-6B\binary_label.xlsx"
train_DATA_PATH_0 = r"D:\Data\Data_NC_State\TU405-6B\Image_Data"

train_label_path_1 = r"D:\Data\Data_UF\Yingbo_LD2_PKYonge_Class1_Mar142019_B\binary_label.xlsx"
train_DATA_PATH_1 = r"D:\Data\Data_UF\Yingbo_LD2_PKYonge_Class1_Mar142019_B\Image_Data"


GENERATE_SQUARE = 64
IMAGE_CHANNELS = 3
CLASS_NUM = 2
BATCH_SIZE = 100
num_training_source = 2
SUB_BATCH_SIZE = int(BATCH_SIZE / num_training_source)
n_samples = int(SUB_BATCH_SIZE / CLASS_NUM)
IMAGE_NUM = 3000
BATCH_NUM = int(IMAGE_NUM / SUB_BATCH_SIZE) + 1
ITERATIONS = 5000


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


print("Start reading training_0 Image & Label data...")

train_list = excel_data(train_label_path_0)

train_list_0 = []
for i, j in enumerate(train_list):
    if j == 0:
        train_list_0.append(i)

train_list_1 = []
for i, j in enumerate(train_list):
    if j == 1:
        train_list_1.append(i)

train_class_0_data_0 = []
for index in range(len(train_list_0)):
    path = os.path.join(train_DATA_PATH_0, str(train_list_0[index]) + ".jpg")
    image = Image.open(path).resize((GENERATE_SQUARE, GENERATE_SQUARE), Image.ANTIALIAS)
    train_class_0_data_0.append(np.asarray(image))

print(len(train_class_0_data_0))

train_class_0_data_0 = np.reshape(train_class_0_data_0, (-1, GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS))


train_class_1_data_0 = []
for index in range(len(train_list_1)):
    path = os.path.join(train_DATA_PATH_0, str(train_list_1[index]) + ".jpg")
    image = Image.open(path).resize((GENERATE_SQUARE, GENERATE_SQUARE), Image.ANTIALIAS)
    train_class_1_data_0.append(np.asarray(image))

print(len(train_class_1_data_0))

train_class_1_data_0 = np.reshape(train_class_1_data_0, (-1, GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS))

train_data_0 = np.concatenate((train_class_0_data_0, train_class_1_data_0), axis=0)

print("train_data_0.shape: ", train_data_0.shape)

###############################################################################################################
print("Start reading training_1 Image & Label data...")

train_list = excel_data(train_label_path_1)

train_list_0 = []
for i, j in enumerate(train_list):
    if j == 0:
        train_list_0.append(i)

train_list_1 = []
for i, j in enumerate(train_list):
    if j == 1:
        train_list_1.append(i)

train_class_0_data_1 = []
for index in range(len(train_list_0)):
    path = os.path.join(train_DATA_PATH_1, str(train_list_0[index]) + ".jpg")
    image = Image.open(path).resize((GENERATE_SQUARE, GENERATE_SQUARE), Image.ANTIALIAS)
    train_class_0_data_1.append(np.asarray(image))

print(len(train_class_0_data_1))

train_class_0_data_1 = np.reshape(train_class_0_data_1, (-1, GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS))

# we can cut some 0 data here to improve training performance

train_class_1_data_1 = []
for index in range(len(train_list_1)):
    path = os.path.join(train_DATA_PATH_1, str(train_list_1[index]) + ".jpg")
    image = Image.open(path).resize((GENERATE_SQUARE, GENERATE_SQUARE), Image.ANTIALIAS)
    train_class_1_data_1.append(np.asarray(image))

print(len(train_class_1_data_1))

train_class_1_data_1 = np.reshape(train_class_1_data_1, (-1, GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS))

train_data_1 = np.concatenate((train_class_0_data_1, train_class_1_data_1), axis=0)

print("train_data_1.shape: ", train_data_1.shape)


########################################################################################################
print("Start reading testing_0 Image & Label data...")

test_list = excel_data(test_label_path_0)

test_list_0 = []
for i, j in enumerate(test_list):
    if j == 0:
        test_list_0.append(i)

test_list_1 = []
for i, j in enumerate(test_list):
    if j == 1:
        test_list_1.append(i)

test_class_0_data_0 = []
for index in range(len(test_list_0)):
    path = os.path.join(test_DATA_PATH_0, str(test_list_0[index]) + ".jpg")
    image = Image.open(path).resize((GENERATE_SQUARE, GENERATE_SQUARE), Image.ANTIALIAS)
    test_class_0_data_0.append(np.asarray(image))

print(len(test_class_0_data_0))

test_class_0_data_0 = np.reshape(test_class_0_data_0, (-1, GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS))

# we can cut some 0 data here to improve training performance

test_class_1_data_0 = []
for index in range(len(test_list_1)):
    path = os.path.join(test_DATA_PATH_0, str(test_list_1[index]) + ".jpg")
    image = Image.open(path).resize((GENERATE_SQUARE, GENERATE_SQUARE), Image.ANTIALIAS)
    test_class_1_data_0.append(np.asarray(image))

print(len(test_class_1_data_0))

test_class_1_data_0 = np.reshape(test_class_1_data_0, (-1, GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS))

########################################################################################################

print("Start reading testing_1 Image & Label data...")

test_list = excel_data(test_label_path_1)

test_list_0 = []
for i, j in enumerate(test_list):
    if j == 0:
        test_list_0.append(i)

test_list_1 = []
for i, j in enumerate(test_list):
    if j == 1:
        test_list_1.append(i)

test_class_0_data_1 = []
for index in range(len(test_list_0)):
    path = os.path.join(test_DATA_PATH_1, str(test_list_0[index]) + ".jpg")
    image = Image.open(path).resize((GENERATE_SQUARE, GENERATE_SQUARE), Image.ANTIALIAS)
    test_class_0_data_1.append(np.asarray(image))

print(len(test_class_0_data_1))

test_class_0_data_1 = np.reshape(test_class_0_data_1, (-1, GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS))

# we can cut some 0 data here to improve training performance

test_class_1_data_1 = []
for index in range(len(test_list_1)):
    path = os.path.join(test_DATA_PATH_1, str(test_list_1[index]) + ".jpg")
    image = Image.open(path).resize((GENERATE_SQUARE, GENERATE_SQUARE), Image.ANTIALIAS)
    test_class_1_data_1.append(np.asarray(image))

print(len(test_class_1_data_1))

test_class_1_data_1 = np.reshape(test_class_1_data_1, (-1, GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS))

test_data = np.concatenate((test_class_0_data_0, test_class_1_data_0, test_class_0_data_1, test_class_1_data_1), axis=0)

a = len(test_class_0_data_0); A = np.zeros((a, 1))
b = len(test_class_1_data_0); B = np.ones((b, 1))
c = len(test_class_0_data_1); C = np.zeros((c, 1))
d = len(test_class_1_data_1); D = np.ones((d, 1))

y_test = np.concatenate((A, B, C, D), axis = 0)
print("Testing data shape: ", test_data.shape)
print("Testing target length: ", len(y_test))

########################################################################################################

print("Start building networks...")

from numpy.random import randn
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
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
c_model.summary()

print("Start training...")

epoch = 0

for i in range(ITERATIONS):
    ####generate supervised real data
    ix_0 = np.random.randint(0, len(train_class_0_data_0), n_samples)
    ix_1 = np.random.randint(0, len(train_class_0_data_1), n_samples)

    X_supervised_samples_class_0_from_source_0 = np.asarray(train_class_0_data_0[ix_0])
    X_supervised_samples_class_0_from_source_1 = np.asarray(train_class_0_data_1[ix_1])
    Y_supervised_samples_class_0 = np.zeros((SUB_BATCH_SIZE, 1))

    ix_0 = np.random.randint(0, len(train_class_1_data_0), n_samples)
    ix_1 = np.random.randint(0, len(train_class_1_data_1), n_samples)
    X_supervised_samples_class_1_from_source_0 = np.asarray(train_class_1_data_0[ix_0])
    X_supervised_samples_class_1_from_source_1 = np.asarray(train_class_1_data_1[ix_1])
    Y_supervised_samples_class_1 = np.ones((SUB_BATCH_SIZE, 1))

    Xsup_real = np.concatenate(
        (X_supervised_samples_class_0_from_source_0,
         X_supervised_samples_class_0_from_source_1,
         X_supervised_samples_class_1_from_source_0,
         X_supervised_samples_class_1_from_source_1), axis=0)
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
