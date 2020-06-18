import cv2
import test
import matplotlib.image as mpimg

input_i = 0
img = cv2.imread('/Users/mac/Desktop/College/Linear Algebra/Project/SegData6_test/Images/' + str(input_i) + '.png')
with open('/Users/mac/Desktop/College/Linear Algebra/Project/SegData6_test/GroundTruthTxt/' + str(input_i) + '.txt', 'r') as file:
    data = file.readlines()

# print(data[0])

contours2points = []
contours4points = []
# data_words = []
for i in range(len(data)):
    # print(data[i][0:4])
    if data[i][0:4] == 'word':
        # data_words.append(data[i].replace('\n', ''))
        splitlines = data[i].split('\t')
        min_x = int(splitlines[1])
        min_y = int(splitlines[2])
        max_x = int(splitlines[3])
        max_y = int(splitlines[4])
        # del data[i]
        # print(data[i])
        # i -= 1
        test.add_contour(contours4points, contours2points, min_x, min_y, max_x, max_y)


imgcont = cv2.drawContours(img.copy(), contours4points, -1, (0, 255, 0), 1)
mpimg.imsave('gtw.png', imgcont, cmap='gray')
