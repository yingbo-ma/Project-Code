#################################################################################################################
import xlrd
import os
from scipy.io import wavfile
import numpy as np

test_label_path = r"D:\Data\Data_UF\Yingbo_LD2_PKYonge_Class1_Mar142019_B\binary_label.xlsx"
test_DATA_PATH = r"D:\Data\Data_UF\Yingbo_LD2_PKYonge_Class1_Mar142019_B\Audio_Data"

train_label_path = r"D:\Data\Data_UF\Jule_LD14_PKYonge_Class1_Mar142019\binary_label.xlsx"
train_DATA_PATH = r"D:\Data\Data_UF\Jule_LD14_PKYonge_Class1_Mar142019\Audio_Data"

GENERATE_SQUARE = 64
IMAGE_CHANNELS = 3
CLASS_NUM = 2
BATCH_SIZE = 100
n_samples = int(BATCH_SIZE / CLASS_NUM)
latent_dim = 100
IMAGE_NUM = 2197
BATCH_NUM = int(IMAGE_NUM / BATCH_SIZE) + 1
ITERATIONS = 3000
num_timesteps = 5
SAMPLE_POINTS = 10000


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


print("Start reading training Audio & Label data...")
train_label_list = excel_data(train_label_path)
train_target_index = [[[i + j] for i in range(num_timesteps)] for j in range(len(train_label_list) - num_timesteps + 1)]
train_target = [[]]

for i in train_target_index:
    temp = []
    temp.append(train_label_list[i[0][0]])
    temp.append(train_label_list[i[1][0]])
    temp.append(train_label_list[i[2][0]])
    temp.append(train_label_list[i[3][0]])
    temp.append(train_label_list[i[4][0]])
    train_target.append(temp)

train_target.remove([])
train_target = np.reshape(train_target, (-1, num_timesteps, 1))
print(train_target.shape)

train_original_data = []

for j in range(len(train_label_list)):
    path = os.path.join(train_DATA_PATH, str(j) + ".wav")
    (sample_rate, sig) = wavfile.read(path)
    data_0 = sig[:, 0]
    ix = np.random.randint(0, len(data_0), SAMPLE_POINTS)
    audio = data_0[ix] / 100
    train_original_data.append(np.asarray(audio))

train_original_data = np.reshape(train_original_data, (-1, SAMPLE_POINTS))


train_data_index = [[[i + j] for i in range(num_timesteps)] for j in range(len(train_original_data) - num_timesteps + 1)]
train_data = [[]]

for k in train_data_index:
    temp = []
    temp.append(train_original_data[k[0][0]])
    temp.append(train_original_data[k[1][0]])
    temp.append(train_original_data[k[2][0]])
    temp.append(train_original_data[k[3][0]])
    temp.append(train_original_data[k[4][0]])
    train_data.append(temp)

train_data.remove([])
train_data = np.reshape(train_data, (-1, num_timesteps, SAMPLE_POINTS))
print("training data shape is: ", train_data.shape)

########################################################################################
print("Start reading testing Audio & Label data...")
test_label_list = excel_data(test_label_path)
test_target_index = [[[i + j] for i in range(num_timesteps)] for j in range(len(test_label_list) - num_timesteps + 1)]
test_target = [[]]

for i in test_target_index:
    temp = []
    temp.append(test_label_list[i[0][0]])
    temp.append(test_label_list[i[1][0]])
    temp.append(test_label_list[i[2][0]])
    temp.append(test_label_list[i[3][0]])
    temp.append(test_label_list[i[4][0]])
    test_target.append(temp)

test_target.remove([])
test_target = np.reshape(test_target, (-1, num_timesteps, 1))
print(test_target.shape)

test_original_data = []

for j in range(len(test_label_list)):
    path = os.path.join(test_DATA_PATH, str(j) + ".wav")
    (sample_rate, sig) = wavfile.read(path)
    data_0 = sig[:, 0]
    ix = np.random.randint(0, len(data_0), SAMPLE_POINTS)
    audio = data_0[ix] / 100
    test_original_data.append(np.asarray(audio))

test_original_data = np.reshape(test_original_data, (-1, SAMPLE_POINTS))


test_data_index = [[[i + j] for i in range(num_timesteps)] for j in range(len(test_original_data) - num_timesteps + 1)]
test_data = [[]]

for k in test_data_index:
    temp = []
    temp.append(test_original_data[k[0][0]])
    temp.append(test_original_data[k[1][0]])
    temp.append(test_original_data[k[2][0]])
    temp.append(test_original_data[k[3][0]])
    temp.append(test_original_data[k[4][0]])
    test_data.append(temp)

test_data.remove([])
test_data = np.reshape(test_data, (-1, num_timesteps, SAMPLE_POINTS))
print("testing data shape is: ", test_data.shape)

########################################################################################
#
from keras.models import Sequential
from keras.layers import Conv2D, LeakyReLU, Dropout, Flatten, TimeDistributed, LSTM, Dense, Input
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.optimizers import Adam
from sklearn.metrics import classification_report

# define cnn model
num_timesteps = 5
TRAIN_PERC = 0.75
BATCH_SIZE = 12
IMAGE_NUM = 2574
BATCH_NUM = int(IMAGE_NUM / BATCH_SIZE) + 1
ITERATIONS = 3000

lstm = Sequential()
lstm.add(LSTM((50), return_sequences=True, input_shape=(num_timesteps, SAMPLE_POINTS)))
lstm.add(Dropout(0.2))
lstm.add(Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01)))
lstm.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
lstm.summary()

######################################################################################################
print("Start training...")
epoch = 0

for i in range(ITERATIONS):

    ix = np.random.randint(0, len(train_data), BATCH_SIZE)
    X_supervised_samples = np.asarray(train_data[ix])
    Y_supervised_samples = np.asarray(train_target[ix])

    # update supervised discriminator (c)
    c_loss, c_acc = lstm.train_on_batch(X_supervised_samples, Y_supervised_samples)

    if (i + 1) % (BATCH_NUM * 1) == 0:
        epoch += 1
        print(f"Epoch {epoch}, c model accuracy on training data: {c_acc}")
        _, test_acc = lstm.evaluate(test_data, test_target, verbose=0)
        print(f"Epoch {epoch}, c model accuracy on test data: {test_acc}")
        y_pred = lstm.predict(test_data, batch_size=60, verbose=0)

        pred_list = y_pred.tolist()

        for i in range(len(pred_list)):
            for j in range(5):
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

        final_pred = np.asarray(final_pred_list)
        print(classification_report(test_label_list, final_pred_list))