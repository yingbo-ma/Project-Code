#################################################################################################################
import xlrd
import os
from scipy.io import wavfile
import numpy as np

test_label_path_1 = r"D:\Data\Data_NC_State\TU409-10B\binary_label.xlsx"
test_DATA_PATH_1 = r"D:\Data\Data_NC_State\TU409-10B\Audio_Data"

train_label_path_1 = r"D:\Data\Data_NC_State\TU405-6B\binary_label.xlsx"
train_DATA_PATH_1 = r"D:\Data\Data_NC_State\TU405-6B\Audio_Data"

test_label_path_0 = r"D:\Data\Data_UF\Jule_LD14_PKYonge_Class1_Mar142019\binary_label.xlsx"
test_DATA_PATH_0 = r"D:\Data\Data_UF\Jule_LD14_PKYonge_Class1_Mar142019\Audio_Data"

train_label_path_0 = r"D:\Data\Data_UF\Yingbo_LD2_PKYonge_Class1_Mar142019_B\binary_label.xlsx"
train_DATA_PATH_0 = r"D:\Data\Data_UF\Yingbo_LD2_PKYonge_Class1_Mar142019_B\Audio_Data"

CLASS_NUM = 2
BATCH_SIZE = 100
n_samples = int(BATCH_SIZE / CLASS_NUM)
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

############################################################################################################################################
print("Start reading training_0 Audio & Label data...")

train_label_list_0 = excel_data(train_label_path_0)
train_target_0_index = [[[i + j] for i in range(num_timesteps)] for j in range(len(train_label_list_0) - num_timesteps + 1)]
train_target_0 = [[]]

for i in train_target_0_index:
    temp = []
    temp.append(train_label_list_0[i[0][0]])
    temp.append(train_label_list_0[i[1][0]])
    temp.append(train_label_list_0[i[2][0]])
    temp.append(train_label_list_0[i[3][0]])
    temp.append(train_label_list_0[i[4][0]])
    train_target_0.append(temp)

train_target_0.remove([])
train_target_0 = np.reshape(train_target_0, (-1, num_timesteps, 1))
print(train_target_0.shape)

train_original_data_0 = []

for j in range(len(train_label_list_0)):
    path = os.path.join(train_DATA_PATH_0, str(j) + ".wav")
    (sample_rate, sig) = wavfile.read(path)
    data_0 = sig[:, 0]
    ix = np.random.randint(0, len(data_0), SAMPLE_POINTS)
    audio = data_0[ix] / 100
    train_original_data_0.append(np.asarray(audio))

train_data_0_index = [[[i + j] for i in range(num_timesteps)] for j in range(len(train_original_data_0) - num_timesteps + 1)]
train_data_0 = [[]]

for k in train_data_0_index:
    temp = []
    temp.append(train_original_data_0[k[0][0]])
    temp.append(train_original_data_0[k[1][0]])
    temp.append(train_original_data_0[k[2][0]])
    temp.append(train_original_data_0[k[3][0]])
    temp.append(train_original_data_0[k[4][0]])
    train_data_0.append(temp)

train_data_0.remove([])
train_data_0 = np.reshape(train_data_0, (-1, num_timesteps, SAMPLE_POINTS))
print(train_data_0.shape)

#########################################################################################################################

print("Start reading training_1 Audio & Label data...")

train_label_list_1 = excel_data(train_label_path_1)
train_target_1_index = [[[i + j] for i in range(num_timesteps)] for j in range(len(train_label_list_1) - num_timesteps + 1)]
train_target_1 = [[]]

for i in train_target_1_index:
    temp = []
    temp.append(train_label_list_1[i[0][0]])
    temp.append(train_label_list_1[i[1][0]])
    temp.append(train_label_list_1[i[2][0]])
    temp.append(train_label_list_1[i[3][0]])
    temp.append(train_label_list_1[i[4][0]])
    train_target_1.append(temp)

train_target_1.remove([])
train_target_1 = np.reshape(train_target_1, (-1, num_timesteps, 1))
print(train_target_1.shape)

train_original_data_1 = []

for j in range(len(train_label_list_1)):
    path = os.path.join(train_DATA_PATH_1, str(j) + ".wav")
    (sample_rate, sig) = wavfile.read(path)
    data_0 = sig[:, 0]
    ix = np.random.randint(0, len(data_0), SAMPLE_POINTS)
    audio = data_0[ix] / 100
    train_original_data_1.append(np.asarray(audio))

train_data_1_index = [[[i + j] for i in range(num_timesteps)] for j in range(len(train_original_data_1) - num_timesteps + 1)]
train_data_1 = [[]]

for k in train_data_1_index:
    temp = []
    temp.append(train_original_data_1[k[0][0]])
    temp.append(train_original_data_1[k[1][0]])
    temp.append(train_original_data_1[k[2][0]])
    temp.append(train_original_data_1[k[3][0]])
    temp.append(train_original_data_1[k[4][0]])
    train_data_1.append(temp)

train_data_1.remove([])
train_data_1 = np.reshape(train_data_1, (-1, num_timesteps, SAMPLE_POINTS))
print(train_data_1.shape)

#########################################################################################################################

print("Start reading testing_0 Audio & Label data...")

test_label_list_0 = excel_data(test_label_path_0)
test_target_0_index = [[[i + j] for i in range(num_timesteps)] for j in range(len(test_label_list_0) - num_timesteps + 1)]
test_target_0 = [[]]

for i in test_target_0_index:
    temp = []
    temp.append(test_label_list_0[i[0][0]])
    temp.append(test_label_list_0[i[1][0]])
    temp.append(test_label_list_0[i[2][0]])
    temp.append(test_label_list_0[i[3][0]])
    temp.append(test_label_list_0[i[4][0]])
    test_target_0.append(temp)

test_target_0.remove([])
test_target_0 = np.reshape(test_target_0, (-1, num_timesteps, 1))
print(test_target_0.shape)

test_original_data_0 = []

for j in range(len(test_label_list_0)):
    path = os.path.join(test_DATA_PATH_0, str(j) + ".wav")
    (sample_rate, sig) = wavfile.read(path)
    data_0 = sig[:, 0]
    ix = np.random.randint(0, len(data_0), SAMPLE_POINTS)
    audio = data_0[ix] / 100
    test_original_data_0.append(np.asarray(audio))

test_data_0_index = [[[i + j] for i in range(num_timesteps)] for j in range(len(test_original_data_0) - num_timesteps + 1)]
test_data_0 = [[]]

for k in test_data_0_index:
    temp = []
    temp.append(test_original_data_0[k[0][0]])
    temp.append(test_original_data_0[k[1][0]])
    temp.append(test_original_data_0[k[2][0]])
    temp.append(test_original_data_0[k[3][0]])
    temp.append(test_original_data_0[k[4][0]])
    test_data_0.append(temp)

test_data_0.remove([])
test_data_0 = np.reshape(test_data_0, (-1, num_timesteps, SAMPLE_POINTS))
print(test_data_0.shape)


#########################################################################################################################

print("Start reading testing_1 Audio & Label data...")

test_label_list_1 = excel_data(test_label_path_1)
test_target_1_index = [[[i + j] for i in range(num_timesteps)] for j in range(len(test_label_list_1) - num_timesteps + 1)]
test_target_1 = [[]]

for i in test_target_1_index:
    temp = []
    temp.append(test_label_list_1[i[0][0]])
    temp.append(test_label_list_1[i[1][0]])
    temp.append(test_label_list_1[i[2][0]])
    temp.append(test_label_list_1[i[3][0]])
    temp.append(test_label_list_1[i[4][0]])
    test_target_1.append(temp)

test_target_1.remove([])
test_target_1 = np.reshape(test_target_1, (-1, num_timesteps, 1))
print(test_target_1.shape)

test_original_data_1 = []

for j in range(len(test_label_list_1)):
    path = os.path.join(test_DATA_PATH_1, str(j) + ".wav")
    (sample_rate, sig) = wavfile.read(path)
    data_0 = sig[:, 0]
    ix = np.random.randint(0, len(data_0), SAMPLE_POINTS)
    audio = data_0[ix] / 100
    test_original_data_1.append(np.asarray(audio))

test_data_1_index = [[[i + j] for i in range(num_timesteps)] for j in range(len(test_original_data_1) - num_timesteps + 1)]
test_data_1 = [[]]

for k in test_data_1_index:
    temp = []
    temp.append(test_original_data_1[k[0][0]])
    temp.append(test_original_data_1[k[1][0]])
    temp.append(test_original_data_1[k[2][0]])
    temp.append(test_original_data_1[k[3][0]])
    temp.append(test_original_data_1[k[4][0]])
    test_data_1.append(temp)

test_data_1.remove([])
test_data_1 = np.reshape(test_data_1, (-1, num_timesteps, SAMPLE_POINTS))
print(test_data_1.shape)

test_data = np.concatenate((test_data_0, test_data_1), axis=0)
test_target = np.concatenate((test_target_0, test_target_1), axis=0)
print("test data shape is: ", test_data.shape)
print("test target shape is: ", test_target.shape)

test_target_list = []
for i in range(len(test_target)):
    test_target_list.append(test_target[i][0][0])

print("test target list shape is: ", len(test_target_list))

#################################################################################################################

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
ITERATIONS = 5000

lstm = Sequential()
lstm.add(LSTM((50), return_sequences=True, input_shape=(num_timesteps, SAMPLE_POINTS)))
lstm.add(Dropout(0.2))
lstm.add(Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01)))
lstm.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
lstm.summary()

######################################################################################################
print("Start training...")
epoch = 0
BATCH_NUM = int((len(train_data_0)+len(train_data_1)) / BATCH_SIZE) + 1

for i in range(ITERATIONS):

    ix_0 = np.random.randint(0, len(train_data_0), int(BATCH_SIZE / 2))
    X_supervised_samples_from_0 = np.asarray(train_data_0[ix_0])
    Y_supervised_samples_from_0 = np.asarray(train_target_0[ix_0])

    ix_1 = np.random.randint(0, len(train_data_1), int(BATCH_SIZE / 2))
    X_supervised_samples_from_1 = np.asarray(train_data_1[ix_1])
    Y_supervised_samples_from_1 = np.asarray(train_target_1[ix_1])

    Xsup_real = np.concatenate(
        (X_supervised_samples_from_0,
         X_supervised_samples_from_1,), axis=0)
    ysup_real = np.concatenate(
        (Y_supervised_samples_from_0,
         Y_supervised_samples_from_1), axis=0)

    # update supervised discriminator (c)
    c_loss, c_acc = lstm.train_on_batch(Xsup_real, ysup_real)

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

        final_pred = np.asarray(final_pred_list)
        print(classification_report(test_target_list, final_pred_list))

