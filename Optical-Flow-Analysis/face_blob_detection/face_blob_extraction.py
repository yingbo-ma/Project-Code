import os
import cv2
import numpy as np

# Define paths
base_dir = os.path.dirname(__file__)
prototxt_path = os.path.join(base_dir + '/model_data/deploy.prototxt')
caffemodel_path = os.path.join(base_dir + '/model_data/weights.caffemodel')

# Read the model
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

if not os.path.exists('faces'):
    print("New directory created")
    os.makedirs('faces')

vc = cv2.VideoCapture("TU405-6B.MP4")
imageIndex = 0

while (vc.isOpened()):

    _, first_frame = vc.read()
    resize_dim = 600
    max_dim = max(first_frame.shape)
    scale = resize_dim / max_dim
    first_frame = cv2.resize(first_frame, None, fx=scale, fy=scale)

    (h, w) = first_frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(first_frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    detections = model.forward()
    print(detections.dtype)

    # Create frame around face
    for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        confidence = detections[0, 0, i, 2]

        # If confidence > 0.5, show box around face
        if (confidence > 0.5):
            cv2.rectangle(first_frame, (startX, startY), (endX, endY), (255, 255, 255), 2)
            first_frame = first_frame[startY:endY, startX:endX]
            print("first frame: ", first_frame)
            print("confidence: ", confidence)
        #     cv2.imwrite(base_dir + '/faces/' + str(imageIndex) + ".jpg", first_frame)
        #     imageIndex += 1
        #     break
            if(first_frame.dtype != None):
                cv2.imwrite(base_dir + '/faces/' + str(imageIndex) + ".jpg", first_frame)
                imageIndex += 1
                savedImage = first_frame
                break
            else:
                cv2.imwrite(base_dir + '/faces/' + str(imageIndex) + ".jpg", savedImage)
                imageIndex += 1
                break
        else:
            print("Nothing is detected!")

    # Read a frame from video
    fps = vc.get(cv2.CAP_PROP_FPS)

    for j in range(int(fps) - 1):
        vc.grab()

    _, next_frame = vc.read()

    first_frame = next_frame

    # Frame are read by intervals of 500 millisecond.
    # The programs breaks out of the while loop when the user presses the ‘q’ key
    if cv2.waitKey(500) & 0xFF == ord('q'):
        break

vc.release()
cv2.destroyAllWindows()