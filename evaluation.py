import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.image as mpimg
from test import seg
from tqdm import tqdm
# from keras.models import Model, load_model

num_of_images = 500
input_images = []
unet_images = []

print('reading input images')
for i in tqdm(range(num_of_images)):
    _img = cv2.imread('/Object-Detection-Metrics-master/CharExtractionEvaluation/Input/' + str(i) + '.png')
    _img_unet = cv2.imread('/Object-Detection-Metrics-master/CharExtractionEvaluation/OurSegmentationOutput/' + str(i) + '.png')
    input_images.append(_img)
    unet_images.append(_img_unet)

'''_contours4points, _contours2points, _morphologically_manipulated, _sub_images, _sub_images_unet = seg(
        input_images[0],
        unet_images[0],
        'word')
imgcont = cv2.drawContours(_img.copy(), _contours4points, -1, (0, 255, 0), 1)
mpimg.imsave('original_segmented.png', imgcont, cmap='gray')
mpimg.imsave('morphologically_manipulated.png', _morphologically_manipulated, cmap='gray')'''


def full_seg_eval(img, img_unet, apply_erosion=True, remove_noise=True):
    eval_txt = []

    contours4points_lines, contours2points_lines, morphologically_manipulated_lines, sub_images_lines, sub_images_unet_lines = seg(
        img,
        img_unet,
        'line', apply_erosion, remove_noise)

    for i in range(len(contours2points_lines)):
        eval_txt.append(add_eval_line('line', contours2points_lines, i))

    contours4points_words, contours2points_words, morphologically_manipulated_words, sub_images_words, sub_images_unet_words = seg(
        img,
        img_unet,
        'word', apply_erosion, remove_noise)

    for j in range(len(sub_images_words)):
        eval_txt.append(add_eval_line('word', contours2points_words, j))

    contours4points_chars, contours2points_chars, morphologically_manipulated_chars, sub_images_chars, sub_images_unet_chars = seg(
        img,
        img_unet,
        'char', apply_erosion, remove_noise)

    for k in range(len(sub_images_chars)):
        eval_txt.append(add_eval_line('char', contours2points_chars, k))

    return eval_txt


def add_eval_line(mode, contours2points, _i):
    return mode + '\t' + str(contours2points[_i][0][0]) + '\t' + str(contours2points[_i][0][1]) + '\t' + str(contours2points[_i][1][0]) + '\t' + str(contours2points[_i][1][1]) + '\n'


print('saving no erosion no noise removal')
for i in tqdm(range(num_of_images)):
    file = open('/Object-Detection-Metrics-master/CharExtractionEvaluation/OurSegmentationOutputTxt(NoErosionNoNoiseRemoval)/' + str(i) + '.txt', 'w')
    file.writelines(full_seg_eval(input_images[i], unet_images[i], False, False))
    file.close()

print('saving erosion with no noise removal')
for i in tqdm(range(num_of_images)):
    file = open('/Object-Detection-Metrics-master/CharExtractionEvaluation/OurSegmentationOutputTxt(NoNoiseRemoval)/' + str(i) + '.txt', 'w')
    file.writelines(full_seg_eval(input_images[i], unet_images[i], True, False))
    file.close()

print('saving erosion AND noise removal')
for i in tqdm(range(num_of_images)):
    file = open('/Object-Detection-Metrics-master/CharExtractionEvaluation/OurSegmentationOutputTxt/' + str(i) + '.txt', 'w')
    file.writelines(full_seg_eval(input_images[i], unet_images[i], True, True))
    file.close()

'''text = ['HI\n', 'Hello\n', 'World\n']
file1 = open('evaluation/OurSegmentationOutputTxt/0.txt', 'w')
file1.writelines(text)
file1.close()'''
