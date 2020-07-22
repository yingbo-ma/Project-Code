from basic_functions import read_excel, data_prepare, train_test_data_split, define_model
import numpy as np
from sklearn.metrics import classification_report

label_path = r"E:\Research Data\Data_UF\Yingbo_LD2_PKYonge_Class1_Mar142019_B\new_label.xlsx"
image_path = r"E:\Research Data\Data_UF\Yingbo_LD2_PKYonge_Class1_Mar142019_B\Image_Data"

PIXEL = 64
IMAGE_CHANNELS = 3
SPLIT_RATIO = 0.75
CLASS_NUM = 3

label_list = read_excel(label_path)
data_0, data_1, data_2 = data_prepare(label_list, image_path, PIXEL, IMAGE_CHANNELS)
train_0_data, train_1_data, train_2_data, train_data, test_data, test_label = train_test_data_split(data_0, data_1, data_2, SPLIT_RATIO)

c_model = define_model(input_shape = (PIXEL, PIXEL, IMAGE_CHANNELS), n_classes = CLASS_NUM)

print("Start training...")

BATCH_SIZE = 60
n_samples = int(BATCH_SIZE / CLASS_NUM)
BATCH_NUM = int(len(train_data) / BATCH_SIZE) + 1
ITERATIONS = 5000

epoch = 0

for i in range(ITERATIONS):

    ix = np.random.randint(0, len(train_0_data), n_samples)
    X_supervised_samples_class_0 = np.asarray(train_0_data[ix])
    Y_supervised_samples_class_0 = np.zeros((n_samples, 1))

    ix = np.random.randint(0, len(train_1_data), n_samples)
    X_supervised_samples_class_1 = np.asarray(train_1_data[ix])
    Y_supervised_samples_class_1 = np.ones((n_samples, 1))

    ix = np.random.randint(0, len(train_2_data), n_samples)
    X_supervised_samples_class_2 = np.asarray(train_2_data[ix])
    Y_supervised_samples_class_2 = 2 * np.ones((n_samples, 1))

    Xsup_real = np.concatenate(
        (X_supervised_samples_class_0, X_supervised_samples_class_1, X_supervised_samples_class_2), axis=0)
    Y_sup_real = np.concatenate(
        (Y_supervised_samples_class_0, Y_supervised_samples_class_1, Y_supervised_samples_class_2), axis=0)

    c_loss, c_acc = c_model.train_on_batch(Xsup_real, Y_sup_real)

    if (i + 1) % (BATCH_NUM * 1) == 0:
        epoch += 1
        print(f"Epoch {epoch}, c model accuracy on training data: {c_acc}")
        _, test_acc = c_model.evaluate(test_data, test_label, verbose=0)
        print(f"Epoch {epoch}, c model accuracy on test data: {test_acc}")
        y_pred = c_model.predict(test_data, batch_size=60, verbose=0)
        y_pred_bool = np.argmax(y_pred, axis=1)
        print(classification_report(test_label, y_pred_bool))
