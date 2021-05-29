'''
Copyrights: ©2021 @Laffery
Date: 2021-05-25 19:55:46
LastEditor: Laffery
LastEditTime: 2021-05-29 11:49:47
'''
from utils import image_filename
import cv2
import numpy as np
import math

def distance(p1, p2):
    sum = 0
    for i in range(0, len(p1)):
        sum += math.pow(p1[i] - p2[i], 2)
    return math.sqrt(sum)

def find_nidus(filename, savename=''):
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
    # print (len(status))

    def is4(n):
        return n[1] == 4

    parenchyma_list = list(filter(is4, np.argwhere((status > 1000) | (status < 2))))
    parenchyma_list = [parenchyma[0] for parenchyma in parenchyma_list]
            
    for i in range(0, len(labels)):
        for j in range(0, len(labels[i])):
            if labels[i, j] in parenchyma_list:
                gray[i, j] = 0
            else:
                gray[i, j] = labels[i, j]          
    
    '''连通域中心坐标''' 
    ls = []

    for i in range(0, len(status)):
        if i not in parenchyma_list:
            x, y, w, h = status[i, :4]
            point = {'label': i, 'xy' : ((int)(x + w/2), (int)(y+h/2)) }
            ls.append(point)
    # print(len(ls))

    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    '''连通域中心距离计算'''
    for i in range(0, len(ls) - 1):
        p1 = ls[i]['xy']
        for j in range(i+1, len(ls)):
            p2 = ls[j]['xy']
            dist = distance(p1, p2)

            '''距离小于，连线'''
            if dist < 5:
                cv2.line(gray, p1, p2, (0, 255, 0))
        # print(distance(ls[i]['xy'], ls[i+1]['xy']))

    cv2.imshow('tmp', gray)

    # element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    # dst = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, element)

    # cv2.imshow('res',dst)

    # cv2.imwrite(savename, gray)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':

    # for i in range(1, 21):
    #     find_nidus(image_filename(0, i, '02'), image_filename(0, i, '03'))
    #     find_nidus(image_filename(1, i, '02'), image_filename(1, i, '03'))

    # print ('Results saved as xx00003.jpg')
    find_nidus(image_filename(0, 1, '02'))