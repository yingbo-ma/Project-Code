print("Start building networks...")

GENERATE_SQUARE = 64
IMAGE_CHANNELS = 3
CLASS_NUM = 2
BATCH_SIZE = 12
n_samples = int(BATCH_SIZE / CLASS_NUM)
ITERATIONS = 3000
num_timesteps = 5
TRAIN_PERC = 0.75

from keras.models import Sequential
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers import Conv2D, LeakyReLU, Dropout, Flatten, TimeDistributed, LSTM, Dense, Input
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.optimizers import Adam
from sklearn.metrics import classification_report

convlstm = Sequential()

convlstm.add(ConvLSTM2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation=LeakyReLU(alpha=0.2),
                        dropout=0.2, input_shape=(None, GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS),
                        padding='same', return_sequences=True))
convlstm.add(BatchNormalization(momentum=0.9))

convlstm.add(ConvLSTM2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation=LeakyReLU(alpha=0.2),
                        dropout=0.2, padding='same', return_sequences=True))
convlstm.add(BatchNormalization(momentum=0.9))

convlstm.add(ConvLSTM2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation=LeakyReLU(alpha=0.2),
                        dropout=0.2, padding='same', return_sequences=True))
convlstm.add(BatchNormalization(momentum=0.9))

convlstm.add(ConvLSTM2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation=LeakyReLU(alpha=0.2),
                        dropout=0.2, padding='same', return_sequences=True))
convlstm.add(BatchNormalization(momentum=0.9))

convlstm.add(TimeDistributed(Flatten(), input_shape=(num_timesteps, GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS)))

convlstm.add(TimeDistributed(Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01))))

convlstm.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])

convlstm.summary()

print("Start training...")

epoch = 0
BATCH_NUM = int(len(X_train) / BATCH_SIZE) + 1

for i in range(ITERATIONS):

    ix = np.random.randint(0, len(X_train), BATCH_SIZE)
    X_supervised_samples = np.asarray(X_train[ix])
    Y_supervised_samples = np.asarray(y_train[ix])

    # update supervised discriminator (c)
    c_loss, c_acc = convlstm.train_on_batch(X_supervised_samples, Y_supervised_samples)

    if (i + 1) % (BATCH_NUM * 1) == 0:
        epoch += 1
        print(f"Epoch {epoch}, c model accuracy on training data: {c_acc}")
        _, test_acc = convlstm.evaluate(X_test, y_test, verbose=0)
        print(f"Epoch {epoch}, c model accuracy on test data: {test_acc}")
        y_pred = convlstm.predict(X_test, batch_size=60, verbose=0)

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
