#################################################################################################################
import xlrd
import os
import numpy as np
from scipy.io import wavfile

num_timesteps = 5
TRAIN_PERC = 0.75

SAMPLE_POINTS = 10000

label_path = r"D:\Publications\SIGDIAL_2020\Data\Data_NC_State\TU409-10B\binary_label.xlsx"
DATA_PATH = r"D:\Publications\SIGDIAL_2020\Data\Data_NC_State\TU409-10B\Audio_Data"

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


print("Start reading Audio & Label data...")

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
    path = os.path.join(DATA_PATH, str(j) + ".wav")
    (sample_rate, sig) = wavfile.read(path)
    data_0 = sig[:, 0]
    ix = np.random.randint(0, len(data_0), SAMPLE_POINTS)
    audio = data_0[ix] / 100
    original_data.append(np.asarray(audio))

original_data = np.reshape(original_data, (-1, SAMPLE_POINTS))


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
data = np.reshape(data, (-1, num_timesteps, SAMPLE_POINTS))
print(data.shape)

########################################################################################

Train_Num = int(len(data) * TRAIN_PERC)

X_train = data[0 : Train_Num]
# X_train = data[0 : 500]
X_test = data[Train_Num : ]

y_train = target[0 : Train_Num]
# y_train = target[0 : 500]
y_test = target[Train_Num : ]

final_test = label_list[Train_Num : ]

print(len(X_train))
print(len(X_test))
print(len(final_test))

########################################################################################

from keras.models import Sequential
from keras.layers import Dropout, LSTM, Dense
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
BATCH_NUM = int(len(X_train) / BATCH_SIZE) + 1

for i in range(ITERATIONS):

    ix = np.random.randint(0, len(X_train), BATCH_SIZE)
    X_supervised_samples = np.asarray(X_train[ix])
    Y_supervised_samples = np.asarray(y_train[ix])

    # update supervised discriminator (c)
    c_loss, c_acc = lstm.train_on_batch(X_supervised_samples, Y_supervised_samples)

    if (i + 1) % (BATCH_NUM * 1) == 0:
        epoch += 1
        print(f"Epoch {epoch}, c model accuracy on training data: {c_acc}")
        _, test_acc = lstm.evaluate(X_test, y_test, verbose=0)
        print(f"Epoch {epoch}, c model accuracy on test data: {test_acc}")
        y_pred = lstm.predict(X_test, batch_size=60, verbose=0)

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
        print(classification_report(final_test, final_pred_list))