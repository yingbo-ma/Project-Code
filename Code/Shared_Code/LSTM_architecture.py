num_timesteps = 5
TRAIN_PERC = 0.75
SAMPLE_POINTS = 10000

from keras.models import Sequential
from keras.layers import Dropout, LSTM, Dense
from keras import regularizers
from keras.optimizers import Adam
from sklearn.metrics import classification_report

# define LSTM model
BATCH_SIZE = 12
AUDIO_NUM = len(X_train) # X_trian is the training data X_test is the test data
BATCH_NUM = int(AUDIO_NUM / BATCH_SIZE) + 1
ITERATIONS = 3000
num_timesteps = 5
SAMPLE_POINTS = 10000

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

    # update supervised model
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