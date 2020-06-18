import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

sub_block_size = 512


# sub_block_size = 2048


def remove_empty(l):
    no_empty = [x for x in l if (x != [])]
    '''for i in no_empty_lines:
        for j in i:
            if i == [[]]:'''
    # for i in range(len(no_empty_lines)):
    return no_empty


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


def get_contours(img, img_unet, sorting, remove_noise=True, mode='char'):
    contours, hierarchy = cv2.findContours(img_unet, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours4points = []
    contours2points = []
    for i in contours:
        min_x = i[i[:, :, 0].argmin()][0][0]
        min_y = i[i[:, :, 1].argmin()][0][1]
        max_x = i[i[:, :, 0].argmax()][0][0]
        max_y = i[i[:, :, 1].argmax()][0][1]

        up_offset = 0
        down_offset = 0
        left_offset = 0
        right_offset = 0
        if mode == 'char':
            up_offset = 5
            down_offset = 4
            left_offset = 3
            right_offset = 2
        elif mode == 'word':
            up_offset = -2
            down_offset = -2
            left_offset = -5
            right_offset = -5
        elif mode == 'line':
            left_offset = -40
            right_offset = -40

        '''if mode == 'line':
            x_offset = -40'''
        '''if mode == 'word':
            y_offset = -2
            x_offset = -5'''

        if min_x > abs(left_offset):
            min_x = clamp(min_x - left_offset, 0, sub_block_size)
        if max_x < sub_block_size - abs(right_offset):
            max_x = clamp(max_x + right_offset, min_x, sub_block_size)
        if min_y > abs(up_offset):
            min_y = clamp(min_y - up_offset, 0, sub_block_size)
        if max_y < sub_block_size - abs(down_offset):
            max_y = clamp(max_y + down_offset, min_y, sub_block_size)

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
        vertical_kernel = np.ones((9, 4), np.uint8)
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


'''l = []
print(l)
l.append([])
l.append([])
l[0].append([])
l[0].append([])
print(l)'''


def full_seg(img, img_unet):
    full = []
    # full_unet = []

    contours4points_lines, contours2points_lines, morphologically_manipulated_lines, sub_images_lines, sub_images_unet_lines = seg(img,
                                                                                                                 img_unet,
                                                                                                                 'line')
    for i in range(len(sub_images_lines)):
        # if sub_images_lines[i].any():  # Noise Removal
        contours4points_words, contours2points_words, morphologically_manipulated_words, sub_images_words, sub_images_unet_words = seg(
            sub_images_lines[i], sub_images_unet_lines[i], 'word')
        full.append([])
        # full_unet.append([])
        for j in range(len(sub_images_words)):
            # if sub_images_words[j].any():  # Noise Removal
            # print(full)
            contours4points_chars, contours2points_chars, morphologically_manipulated_chars, sub_images_chars, sub_images_unet_chars = seg(
                sub_images_words[j], sub_images_unet_words[j], 'char')
            full[i].append([])
            # full_unet[i].append([])
            for k in range(len(sub_images_chars)):
                # if sub_images_chars[k].any():   # Noise Removal
                full[i][j].append(sub_images_chars[k])
                # full_unet[i][j].append(sub_images_unet_chars[k])

    return full


'''for i in range(len(full)):
    for j in range(len(full[i])):
        full[i] = remove_empty(full[i])
return remove_empty(full)'''


_img = cv2.imread('live.png')
_img1ch = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)

_img_unet = cv2.imread('live_unet.png')
_img_unet_1ch = cv2.cvtColor(_img_unet, cv2.COLOR_BGR2GRAY)
_ret, _thresh = cv2.threshold(_img_unet_1ch, 127, 255, 0)
_kernel = np.ones((8, 3), np.uint8)
eroded = cv2.erode(_thresh, _kernel, iterations=1)
contours, hierarchy = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

_contours4points, _contours2points, _morphologically_manipulated, _sub_images, _sub_images_unet = seg(_img, _img_unet,
                                                                                                      'word')

'''Nr = 1
Nc = len(sub_images_words)
cmap = "gray"

fig, axs = plt.subplots(Nr, Nc)
fig.suptitle('words')

for i in range(Nr):
    for j in range(Nc):
        im = sub_images_words[j]
        axs[j].imshow(im, cmap=cmap)
        # axs[i, j].label_outer()
plt.show()'''

# print(np.array(contours2points), '\n\n---------\n\n', np.array(sort_contours(contours2points)))

'''test = img1ch[min_x:max_x + 1, min_y:max_y + 1]'''

# img3ch = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB)
# print(len(sub_images))
'''imgcont = cv2.drawContours(_img.copy(), _contours4points, -1, (0, 255, 0), 1)
mpimg.imsave('original_segmented.png', imgcont, cmap='gray')
mpimg.imsave('morphologically_manipulated.png', _morphologically_manipulated, cmap='gray')'''

'''mpimg.imsave('individual_contour.png', sub_images, cmap='gray')
mpimg.imsave('individual_contour_unet.png', sub_images, cmap='gray')'''

_full = full_seg(_img, _img_unet)

print(len(_full))
plt.imshow(_full[0][1][4], cmap='gray')

'''for i in range(len(_full)):
    for j in range(len(_full[i])):
        for k in range(len(_full[i][j])):
            mpimg.imsave('live/' + str(i) + '_' + str(j) + '_' + str(k) + '.png', _full[i][j][k])
'''
plt.show()
