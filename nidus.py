'''
Copyrights: ©2021 @Laffery
Date: 2021-05-25 19:55:46
LastEditor: Laffery
LastEditTime: 2021-06-01 18:55:59
'''
from utils import image_filename
import cv2
import numpy as np
import math

def find_nidus(filename, savename='', show=False):
    src = cv2.imread(filename)
    width = src.shape[1]
    height = src.shape[0]

    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

    for raw in range(height):
        for i in range(width):
            if gray[raw, i] > 10:
                gray[raw, i] = 255 if gray[raw, i] <= 108 else 0

    _, labels, status, _ = cv2.connectedComponentsWithStats(gray)

    def is4(n):
        return n[1] == 4

    parenchyma_list = list(filter(is4, np.argwhere((status > 1000) | (status < 2))))
    parenchyma_list = [parenchyma[0] for parenchyma in parenchyma_list]
            
    for i in range(0, len(labels)):
        for j in range(0, len(labels[i])):
            if labels[i, j] in parenchyma_list:
                gray[i, j] = 0

    if show:
        cv2.imshow('res', gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.imwrite(savename, gray)


if __name__ == '__main__':
    for i in range(1, 21):
        find_nidus(image_filename(0, i, '02'), image_filename(0, i, '03'))
        find_nidus(image_filename(1, i, '02'), image_filename(1, i, '03'))

    print ('Results saved as xx00003.jpg')
    # find_nidus(image_filename(1, 1, '02'), image_filename(1, 1, '03'), True)