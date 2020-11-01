# Parameters
vocabulary_size = 1818
num_labels = 9
max_utterance_len = 30
batch_size = 200
hidden_layer = 128
learning_rate = 0.001
num_epoch = 10

from keras.models import Sequential
from keras.layers import Conv2D, LeakyReLU, Dropout, Flatten, TimeDistributed, LSTM, Dense, Input, Bidirectional, Embedding, GlobalMaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.optimizers import Adam, RMSprop
from sklearn.metrics import classification_report

print("------------------------------------")
print("Using parameters...")
print("Vocabulary size: ", vocabulary_size)
print("Number of labels: ", num_labels)
print("Batch size: ", batch_size)
print("Hidden layer size: ", hidden_layer)
print("learning rate: ", learning_rate)
print("Epochs: ", num_epoch)

# Build the model
print("------------------------------------")
print('Build model...')
model = Sequential()
model.add(Embedding(vocabulary_size, 300, input_length=max_utterance_len, mask_zero=False))
model.add(LSTM(hidden_layer, dropout=0.3, return_sequences=True, kernel_initializer='random_uniform', recurrent_initializer='glorot_uniform'))
model.add(TimeDistributed(Dense(hidden_layer, input_shape=(max_utterance_len, hidden_layer)), input_shape=(5, hidden_layer)))
model.add(GlobalMaxPooling1D())
model.add(Dense(num_labels, activation='softmax'))

optimizer = RMSprop(lr=learning_rate, decay=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
print(model.summary())