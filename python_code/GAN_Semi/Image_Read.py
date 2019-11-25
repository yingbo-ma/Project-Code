from PIL import Image
import numpy as np

# Training Data Preprocessing
training_data = []
for filename in tqdm(os.listdir(DATA_PATH)):
    path = os.path.join(DATA_PATH,filename)
    image = Image.open(path).resize((GENERATE_SQUARE,GENERATE_SQUARE),Image.ANTIALIAS)
    training_data.append(np.asarray(image))

training_data = np.reshape(training_data,(-1,GENERATE_SQUARE,GENERATE_SQUARE,IMAGE_CHANNELS))

image = Image.open("/home/yingbo/Desktop/All_Dataset/0.jpg").resize((496,496),Image.ANTIALIAS)
image.show()

I = np.asarray(image)
print(I.shape)