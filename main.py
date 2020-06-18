import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from skimage import morphology

sub_block_size = 512


def remove_empty(l):
    no_empty_lines = [x for x in l if (x != [[]])]
    '''for i in no_empty_lines:
        for j in i:
            if i == [[]]:'''
    return no_empty_lines


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    if boundingBoxes == []:
        return []

    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts)


def clamp(my_value, min_value, max_value):
    return max(min(my_value, max_value), min_value)


def get_contours(img, img_unet, sorting, remove_noise=True):
    contours, hierarchy = cv2.findContours(img_unet, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours4points = []
    contours2points = []
    for i in contours:
        min_x = i[i[:, :, 0].argmin()][0][0]
        min_y = i[i[:, :, 1].argmin()][0][1]
        max_x = i[i[:, :, 0].argmax()][0][0]
        max_y = i[i[:, :, 1].argmax()][0][1]

        y_offset = 1
        x_offset = 1

        # if min_x != 0:
        min_x = clamp(min_x - x_offset, 0, sub_block_size)
        # if max_x != sub_block_size:
        max_x = clamp(max_x + x_offset, min_x, sub_block_size)
        # if min_y != 0:
        min_y = clamp(min_y - y_offset, 0, sub_block_size)
        # if max_y != sub_block_size:
        max_y = clamp(max_y + y_offset, min_y, sub_block_size)

        current_contour = img[min_y:max_y, min_x:max_x]
        if remove_noise:
            if current_contour.any():
                add_contour(contours4points, contours2points, min_x, min_y, max_x, max_y)
        else:
            add_contour(contours4points, contours2points, min_x, min_y, max_x, max_y)

    return contours4points, sort_contours(contours2points, sorting)


def add_contour(contours4points, contours2points, min_x, min_y, max_x, max_y):
    c2 = np.array([[min_x, min_y],
                   [max_x, max_y]])
    contours2points.append(c2)

    c4 = np.array([[min_x, min_y],
                   [max_x, min_y],
                   [max_x, max_y],
                   [min_x, max_y]])
    contours4points.append(c4)


def get_sub_images(img1ch, contours2points):
    sub_images = []
    for i in contours2points:
        min_x = i[0][0]
        max_x = i[1][0]
        min_y = i[0][1]
        max_y = i[1][1]
        '''print(contours2points[i], '|||', (min_x, min_y), (max_x, max_y), '\n')
        print(img1ch.shape)'''
        sub_images.append(img1ch[min_y:max_y + 1, min_x:max_x + 1])

    return sub_images


def seg(img, img_unet, mode, apply_erosion=True, remove_noise=True):
    if img.ndim == 3:
        img1ch = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img1ch = img

    if img_unet.ndim == 3:
        img1ch_unet = cv2.cvtColor(img_unet, cv2.COLOR_BGR2GRAY)
    else:
        img1ch_unet = img_unet

    ret, thresh = cv2.threshold(img1ch_unet, 127, 255, 0)

    sorting = 'left-to-right'

    # if mode == 'char':
    morphologically_manipulated = thresh

    if apply_erosion:
        vertical_kernel = np.ones((8, 3), np.uint8)
        morphologically_manipulated = cv2.erode(thresh, vertical_kernel, iterations=1)

    if mode == 'line':
        horizontal_kernel = np.ones((1, 80), np.uint8)
        morphologically_manipulated = cv2.dilate(thresh, horizontal_kernel, iterations=1)
        sorting = 'top-to-bottom'
    elif mode == 'word':
        kernel = np.ones((5, 10), np.uint8)
        morphologically_manipulated = cv2.dilate(thresh, kernel, iterations=1)

    contours4points, contours2points = get_contours(img1ch, morphologically_manipulated, sorting, remove_noise, mode)

    return contours4points, \
           contours2points, \
           morphologically_manipulated, \
           get_sub_images(img1ch, contours2points), \
           get_sub_images(thresh, contours2points)



def full_seg(img, img_unet):
    full = []
    # full_unet = []

    contours4points, contours2points, morphologically_manipulated, sub_images_lines, sub_images_unet_lines = seg(img,
                                                                                                                 img_unet,
                                                                                                                 'line')
    for i in range(len(sub_images_lines)):
        contours4points, contours2points, morphologically_manipulated, sub_images_words, sub_images_unet_words = seg(
            sub_images_lines[i], sub_images_unet_lines[i], 'word')
        full.append([])
        # full_unet.append([])
        for j in range(len(sub_images_words)):
            # print(full)
            contours4points, contours2points, morphologically_manipulated, sub_images_chars, sub_images_unet_chars = seg(
                sub_images_words[j], sub_images_unet_words[j], 'char')
            full[i].append([])
            # full_unet[i].append([])
            for k in range(len(sub_images_chars)):
                full[i][j].append(sub_images_chars[k])
                # full_unet[i][j].append(sub_images_unet_chars[k])

    return remove_empty(full)