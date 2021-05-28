from utils import image_filename
import cv2
import numpy as np

def find_nidus(filename, savename):
    src = cv2.imread(filename)
    width = src.shape[1]
    height = src.shape[0]

    # cv2.imshow('src', src)

    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    # _, gray = cv2.threshold(src, 108, 255, cv2.THRESH_BINARY)

    for raw in range(height):
        for i in range(width):
            if gray[raw, i] > 10:
                gray[raw, i] = 255 if gray[raw, i] <= 108 else 0

    _, labels, status, _ = cv2.connectedComponentsWithStats(gray)
    # print (status)

    def is4(n):
        return n[1] == 4

    parenchyma_list = list(filter(is4, np.argwhere((status > 6000))))
    parenchyma_list = [parenchyma[0] for parenchyma in parenchyma_list]

    for i in range(0, len(labels)):
        for j in range(0, len(labels[i])):
            if labels[i, j] in parenchyma_list:
                gray[i, j] = 0

    # cv2.imshow('tmp', gray)

    cv2.imwrite(savename, gray)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':

    for i in range(1, 21):
        find_nidus(image_filename(0, i, '02'), image_filename(0, i, '03'))
        find_nidus(image_filename(1, i, '02'), image_filename(1, i, '03'))

    print ('Results saved as xx00003.jpg')
    # find_nidus(image_filename(0, 2, '02'))