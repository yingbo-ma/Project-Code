from refined_basic_functions import read_excel, spectrogram_data_prepare, optical_flow_data_prepare, \
    multiple_inputs_model
import numpy as np
from sklearn.metrics import classification_report

label_path = r"E:\Research Data\Data_UF\Yingbo_LD2_PKYonge_Class1_Mar142019_B\new_label.xlsx"
spectrogram_image_path = r"E:\Research Data\Data_UF\Yingbo_LD2_PKYonge_Class1_Mar142019_B\Image_Data"
left_face_optical_flow_image_path = r"E:\Research Code\Optical-Flow-Analysis\face_blob_detection\left_faces_optical_flow"
right_face_optical_flow_image_path = r"E:\Research Code\Optical-Flow-Analysis\face_blob_detection\right_faces_optical_flow"

PIXEL = 64
SPECTRO_IMAGE_CHANNELS = 3
OPTIC_IMAGE_CHANNELS = 1
SPLIT_RATIO = 0.75
CLASS_NUM = 3
OPTIC_CLASS_NUM = 1

label_list = read_excel(label_path)
print("Reading spectrogram data...")
spectrogram_data_0_train, spectrogram_data_1_train, spectrogram_data_2_train, spectrogram_train_data, spectrogram_test_data, test_label, shuffled_list_0, shuffled_list_1, shuffled_list_2 = spectrogram_data_prepare(
    label_list, spectrogram_image_path, PIXEL, SPECTRO_IMAGE_CHANNELS, SPLIT_RATIO)
print("Reading left face motion data...")
left_optical_flow_data_0_train, left_optical_flow_data_1_train, left_optical_flow_data_2_train, left_optical_flow_train_data, left_optical_flow_test_data = optical_flow_data_prepare(
    label_list, left_face_optical_flow_image_path, PIXEL, OPTIC_IMAGE_CHANNELS, SPLIT_RATIO, shuffled_list_0,
    shuffled_list_1, shuffled_list_2)
print("Reading right face motion data...")
right_optical_flow_data_0_train, right_optical_flow_data_1_train, right_optical_flow_data_2_train, right_optical_flow_train_data, right_optical_flow_test_data = optical_flow_data_prepare(
    label_list, right_face_optical_flow_image_path, PIXEL, OPTIC_IMAGE_CHANNELS, SPLIT_RATIO, shuffled_list_0,
    shuffled_list_1, shuffled_list_2)

print("Building model...")
c_model = multiple_inputs_model(input_shape_A=(PIXEL, PIXEL, SPECTRO_IMAGE_CHANNELS),
                                input_shape_B=(PIXEL, PIXEL, OPTIC_IMAGE_CHANNELS),
                                input_shape_C=(PIXEL, PIXEL, OPTIC_IMAGE_CHANNELS), n_classes=CLASS_NUM)

print("Start training...")

BATCH_SIZE = 270
n_samples = int(BATCH_SIZE / CLASS_NUM)
BATCH_NUM = int(len(spectrogram_train_data) / BATCH_SIZE) + 1
ITERATIONS = 5000

epoch = 0

for i in range(ITERATIONS):

    ix = np.random.randint(0, len(spectrogram_data_0_train), n_samples)

    # prepare training data_0 for acoustic model
    Acoustic_X_supervised_samples_class_0 = np.asarray(spectrogram_data_0_train[ix])
    List_Acoustic_Y_supervised_samples = [[1, 0, 0]]
    for index in range(n_samples - 1):
        List_Acoustic_Y_supervised_samples.append([1, 0, 0])

    # prepare training data_0 for left_motion model
    left_Motion_X_supervised_samples_class_0 = np.asarray(left_optical_flow_data_0_train[ix])

    # prepare training data_0 for right_motion model
    right_Motion_X_supervised_samples_class_0 = np.asarray(right_optical_flow_data_0_train[ix])

    ix = np.random.randint(0, len(spectrogram_data_1_train), n_samples)

    # prepare training data_1 for acoustic model
    Acoustic_X_supervised_samples_class_1 = np.asarray(spectrogram_data_1_train[ix])
    for index in range(n_samples):
        List_Acoustic_Y_supervised_samples.append([0, 1, 0])

    # prepare training data_1 for left_motion model
    left_Motion_X_supervised_samples_class_1 = np.asarray(left_optical_flow_data_1_train[ix])

    # prepare training data_1 for right_motion model
    right_Motion_X_supervised_samples_class_1 = np.asarray(right_optical_flow_data_1_train[ix])

    ix = np.random.randint(0, len(spectrogram_data_2_train), n_samples)

    # prepare training data_2 for acoustic model
    Acoustic_X_supervised_samples_class_2 = np.asarray(spectrogram_data_2_train[ix])
    for index in range(n_samples):
        List_Acoustic_Y_supervised_samples.append([0, 0, 1])

    # prepare training data_2 for left_motion model
    left_Motion_X_supervised_samples_class_2 = np.asarray(left_optical_flow_data_2_train[ix])

    # prepare training data_2 for right_motion model
    right_Motion_X_supervised_samples_class_2 = np.asarray(right_optical_flow_data_2_train[ix])

    # concatenate acoustic training data

    Acoustic_Xsup_real = np.concatenate(
        (Acoustic_X_supervised_samples_class_0, Acoustic_X_supervised_samples_class_1,
         Acoustic_X_supervised_samples_class_2), axis=0)
    Y_sup_real = np.asarray(List_Acoustic_Y_supervised_samples)

    # concatenate left_motion training data

    left_Motion_Xsup_real = np.concatenate(
        (left_Motion_X_supervised_samples_class_0, left_Motion_X_supervised_samples_class_1,
         left_Motion_X_supervised_samples_class_2), axis=0)

    # concatenate right_motion training data

    right_Motion_Xsup_real = np.concatenate(
        (right_Motion_X_supervised_samples_class_0, right_Motion_X_supervised_samples_class_1,
         right_Motion_X_supervised_samples_class_2), axis=0)

    # update model
    c_loss, c_acc = c_model.train_on_batch(x=[Acoustic_Xsup_real, left_Motion_Xsup_real, right_Motion_Xsup_real],
                                           y=Y_sup_real)

    if (i + 1) % (BATCH_NUM * 1) == 0:
        epoch += 1
        print(f"Epoch {epoch}, acoustic model accuracy on training data: {c_acc}")
        _, test_acc = c_model.evaluate(
            [spectrogram_test_data, left_optical_flow_test_data, right_optical_flow_test_data], test_label, verbose=0)
        print(f"Epoch {epoch}, c model accuracy on test data: {test_acc}")
        y_pred = c_model.predict([spectrogram_test_data, left_optical_flow_test_data, right_optical_flow_test_data],
                                 batch_size=60, verbose=0)
        y_pred_bool = np.argmax(y_pred, axis=1)
        test_label_bool = np.argmax(test_label, axis=1)
        print(classification_report(test_label_bool, y_pred_bool))