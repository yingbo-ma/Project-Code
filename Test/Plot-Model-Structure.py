from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras import regularizers
from keras.utils.vis_utils import plot_model

from keras.models import Sequential
from keras.layers import Conv2D, LeakyReLU, Dropout, Flatten, TimeDistributed, LSTM, Dense, Input
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.optimizers import Adam

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

cnn_lstm.add(TimeDistributed(cnn, input_shape=(5, 64, 64, 3)))
cnn_lstm.add(LSTM((50), return_sequences=True))
cnn_lstm.add(Dropout(0.2))
cnn_lstm.add(Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01)))

cnn_lstm.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])

cnn_lstm.summary()

plot_model(cnn_lstm, to_file='model_plot.png', show_shapes=True, show_layer_names=True)



