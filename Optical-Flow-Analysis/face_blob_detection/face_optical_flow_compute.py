import cv2
import numpy as np
import os

base_dir = os.path.dirname(__file__)

if not os.path.exists('left_faces_optical_flow'):
    print("New directory created")
    os.makedirs('left_faces_optical_flow')

if not os.path.exists('right_faces_optical_flow'):
    print("New directory created")
    os.makedirs('right_faces_optical_flow')

image_folder = base_dir + "/right_faces/"
list = os.listdir(image_folder) # dir is your directory path

first_image_path = image_folder + "0.jpg"
first_image = cv2.imread(first_image_path)

def imageResize(inputImage):
    resize_dim = 600
    dim = (resize_dim, resize_dim)
    return cv2.resize(inputImage, dim, interpolation = cv2.INTER_AREA)

first_image = imageResize(first_image)
print(first_image.shape)
first_gray = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)

mask = np.zeros_like(first_image)
mask[..., 1] = 255

for index in range(len(list)):
    next_image = cv2.imread(image_folder + str(index + 1) + ".jpg")
    next_image = imageResize(next_image)
    next_gray = cv2.cvtColor(next_image, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(first_gray, next_gray, None, pyr_scale=0.5, levels=5, winsize=11, iterations=5, poly_n=5, poly_sigma=1.1, flags=0)

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    mask[..., 0] = angle * 180 / np.pi / 2
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # The optical flow field color-coding. Smaller vectors are lighter and color represents the direction
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    # cv2.imshow("Dense optical flow", rgb)
    cv2.imwrite(base_dir + '/right_faces_optical_flow/' + str(index) + ".jpg", magnitude)

    first_gray = next_gray
    if cv2.waitKey(500) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()