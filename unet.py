import numpy as np
from tensorflow.keras.models import load_model
from tqdm import tqdm
from keras_tqdm import TQDMCallback, TQDMNotebookCallback
import cv2
import imageio
import matplotlib.pyplot as plt

our_unet_model = load_model('trained_on_separated_detol_99.h5')

'''input_images = []
start = 0
end = 500

print('loading:')
for i in tqdm(range(start, end)):
    img = cv2.imread('/Users/mac/Desktop/College/Linear Algebra/Project/CharExtraction/Object-Detection-Metrics-master/CharExtractionEvaluation/input/' + str(i) + '.png')
    # print(img)
    img1ch = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    img1ch = img1ch/255.0
    # img1ch.astype('float')
    input_images.append(img1ch)

length = len(input_images)
input_images = np.array(input_images)
# print(input_images.shape)
input_images = input_images.reshape((length, 512, 512, 1))

print('predicting:')
out = our_unet_model.predict(input_images)

print('saving:')
for i in tqdm(range(len(out))):
    imageio.imsave('/Users/mac/Desktop/College/Linear Algebra/Project/CharExtraction/Object-Detection-Metrics-master/CharExtractionEvaluation/OurSegmentationOutput/' + str(i+start) + '.png', (out[i]*255).astype('uint8'))
'''
'''plt.imshow(out[0].reshape((512, 512)), cmap='gray')
plt.show()'''

img = cv2.imread('live2.png')
img1ch = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
img1ch = img1ch/255.0
img1ch = np.array(img1ch)
img1ch = img1ch.reshape((1, 512, 512, 1))
out = our_unet_model.predict(img1ch)
imageio.imsave('live2_unet.png', (out[0]*255).astype('uint8'))
# print(out)
