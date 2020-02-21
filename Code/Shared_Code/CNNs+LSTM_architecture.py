from keras.models import Sequential
from keras.layers import Conv2D, LeakyReLU, Dropout, Flatten, TimeDistributed, LSTM, Dense, Input
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.optimizers import Adam
from sklearn.metrics import classification_report

cnn = Sequential()
cnn.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
cnn.add(BatchNormalization(momentum=0.9))
cnn.add(LeakyReLU(alpha=0.2))
cnn.add(Dropout(0.2))

cnn.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
cnn.add(BatchNormalization(momentum=0.9))
cnn.add(LeakyReLU(alpha=0.2))
cnn.add(Dropout(0.2))

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
cnn_lstm.add(LSTM((50), return_sequences=True))
cnn_lstm.add(Dropout(0.2))
cnn_lstm.add(Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01)))

cnn_lstm.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])

cnn_lstm.summary()

######################################################################################################
print("Start training...")
epoch = 0

for i in range(ITERATIONS):

    ix = np.random.randint(0, len(train_data), BATCH_SIZE)
    X_supervised_samples = np.asarray(train_data[ix])
    Y_supervised_samples = np.asarray(train_target[ix])

    # update supervised discriminator (c)
    c_loss, c_acc = cnn_lstm.train_on_batch(X_supervised_samples, Y_supervised_samples)

    if (i + 1) % (BATCH_NUM * 1) == 0:
        epoch += 1
        print(f"Epoch {epoch}, c model accuracy on training data: {c_acc}")
        _, test_acc = cnn_lstm.evaluate(test_data, test_target, verbose=0)
        print(f"Epoch {epoch}, c model accuracy on test data: {test_acc}")
        y_pred = cnn_lstm.predict(test_data, batch_size=60, verbose=0)

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