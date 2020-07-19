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


# define model
def define_discriminator(in_shape=(64, 64, 3), n_classes=2):
    # image input
    in_image = Input(shape=in_shape)

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
    ix = np.random.randint(0, len(train_class_0_data), n_samples)
    X_supervised_samples_class_0 = np.asarray(train_class_0_data[ix])
    Y_supervised_samples_class_0 = np.zeros((n_samples, 1))

    ix = np.random.randint(0, len(train_class_1_data), n_samples)
    X_supervised_samples_class_1 = np.asarray(train_class_1_data[ix])
    Y_supervised_samples_class_1 = np.ones((n_samples, 1))

    Xsup_real = np.concatenate(
        (X_supervised_samples_class_0, X_supervised_samples_class_1), axis=0)
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
