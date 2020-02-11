from keras.models import Sequential
from keras.layers import Conv2D, LeakyReLU, Dropout, Flatten, TimeDistributed, LSTM, Dense, Input
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.optimizers import Adam
from sklearn.metrics import classification_report

# define cnn model
GENERATE_SQUARE = 64
IMAGE_CHANNELS = 3
in_shape = (GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS)
num_timesteps = 5
TRAIN_PERC = 0.75
BATCH_SIZE = 12
IMAGE_NUM = 2574
BATCH_NUM = int(IMAGE_NUM / BATCH_SIZE) + 1
ITERATIONS = 2000

cnn = Sequential()
cnn.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
cnn.add(BatchNormalization(momentum=0.9))
cnn.add(LeakyReLU(alpha=0.2))
cnn.add(Dropout(0.2))

cnn.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
cnn.add(BatchNormalization(momentum=0.9))
cnn.add(LeakyReLU(alpha=0.2))
cnn.add(Dropout(0.2))

cnn.add(Flatten())

cnn_lstm = Sequential()

cnn_lstm.add(TimeDistributed(cnn, input_shape=(num_timesteps, GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS)))
cnn_lstm.add(LSTM((num_timesteps), return_sequences=True))
cnn_lstm.add(Dropout(0.2))
cnn_lstm.add(LSTM((num_timesteps), return_sequences=True))
cnn_lstm.add(Dropout(0.2))
cnn_lstm.add(Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01)))

cnn_lstm.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])

cnn_lstm.summary()

#################################################################################################################
import xlrd
import os
from PIL import Image
import numpy as np

label_path = r"D:\Data\LD2_PKYonge_Class1_Mar142019_B\binary_label.xlsx"
DATA_PATH = r"D:\Data\LD2_PKYonge_Class1_Mar142019_B\Image_Data"

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
target_index = [[[i + j] for i in range(num_timesteps)] for j in range(len(label_list) - num_timesteps + 1)]
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
target = np.reshape(target, (-1, num_timesteps, 1))
print(target.shape)


original_data = []

for j in range(len(label_list)):
    path = os.path.join(DATA_PATH, str(j) + ".jpg")
    image = Image.open(path).resize((GENERATE_SQUARE, GENERATE_SQUARE), Image.ANTIALIAS)
    original_data.append(np.asarray(image))

original_data = np.reshape(original_data, (-1, GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS))

data_index = [[[i + j] for i in range(num_timesteps)] for j in range(len(original_data) - num_timesteps + 1)]
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
data = np.reshape(data, (-1, num_timesteps, GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS))
print(data.shape)

########################################################################################

Train_Num = int(len(data) * TRAIN_PERC)

X_train = data[0 : Train_Num]
X_test = data[Train_Num : ]

y_train = target[0 : Train_Num]
y_test = target[Train_Num : ]

final_test = label_list[Train_Num : ]

########################################################################################

print("Start training...")

epoch = 0

for i in range(ITERATIONS):

    ix = np.random.randint(0, len(X_train), BATCH_SIZE)
    X_supervised_samples = np.asarray(X_train[ix])
    Y_supervised_samples = np.asarray(y_train[ix])

    # update supervised discriminator (c)
    c_loss, c_acc = cnn_lstm.train_on_batch(X_supervised_samples, Y_supervised_samples)

    if (i + 1) % (BATCH_NUM * 1) == 0:
        epoch += 1
        print(f"Epoch {epoch}, c model accuracy on training data: {c_acc}")
        _, test_acc = cnn_lstm.evaluate(X_test, y_test, verbose=0)
        print(f"Epoch {epoch}, c model accuracy on test data: {test_acc}")
        y_pred = cnn_lstm.predict(X_test, batch_size=60, verbose=0)

        pred_list = y_pred.tolist()

        for i in range(len(pred_list)):
            for j in range(num_timesteps):
                if pred_list[i][j] > [0.5]:
                    pred_list[i][j] = [1]
                else:
                    pred_list[i][j] = [0]

        final_pred_list = []
        for i in range(len(pred_list)):
            final_pred_list.append(pred_list[i][0][0])
        final_pred_list.append(pred_list[i][1][0])
        final_pred_list.append(pred_list[i][2][0])
        final_pred_list.append(pred_list[i][3][0])
        final_pred_list.append(pred_list[i][4][0])

        print(final_pred_list)
        print(final_test)

        final_pred = np.asarray(final_pred_list)
        print(classification_report(final_test, final_pred_list))