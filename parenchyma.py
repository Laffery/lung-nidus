'''
Copyrights: ©2021 @Laffery
Date: 2021-05-25 19:38:29
LastEditor: Laffery
LastEditTime: 2021-05-29 10:54:17
'''
from utils import image_filename
import cv2
import numpy as np

WHITE = [255, 255, 255]
BLACK = [0, 0, 0]

def recognize_parenchyma(filename, savename):
    '''
    分割图片中的肺实质（lung parenchyma）\\
    filename 为图片文件名
    '''
    src = cv2.imread(filename)
    width = src.shape[1]
    height = src.shape[0]

    res = src
    scale = 5
    offset = (int)(height / scale)

    '''去除上下1/5的图片内容（文字）'''
    for raw in range(height):
        if raw <= offset or raw >= (scale - 1) * offset:
            res[raw] = [ BLACK ] * width

    '''转为灰度图，反阈值二值化'''
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, src = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    
    '''连通域'''
    _, labels, status, _ = cv2.connectedComponentsWithStats(src)

    def is4(n):
        '''一个辅助函数，用以判断是否为status第四列'''
        return n[1] == 4

    '''过滤连通域，获得其中面积在400~2000的连通域'''
    trachea_list = list(filter(is4, np.argwhere((status <= 2000) & (status >= 400))))
    trachea_list = [trachea[0] for trachea in trachea_list]

    '''对应连通域像素点剔除 '''
    if len(trachea_list):
        for i in range(0, height):
            for j in range(0, width):
                if(labels[i, j] in trachea_list):
                    src[i, j] = 0

    '''闭操作填充细小孔洞'''
    src = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    dst = cv2.morphologyEx(src, cv2.MORPH_CLOSE, element)
    
    '''reticular目录下的图片可能会存在中间的大圆白，本部分内容保存该圆洞'''
    dst = cv2.bitwise_not(dst)

    gray = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
    _, labels, status, _ = cv2.connectedComponentsWithStats(gray)

    parenchyma_list = list(filter(is4, np.argwhere(status < 6000)))
    parenchyma_list = [parenchyma[0] for parenchyma in parenchyma_list]

    if len(parenchyma_list):
        for i in range(0, len(labels)):
            for j in range(0, len(labels[i])):
                if labels[i, j] in parenchyma_list:
                    dst[i, j] = BLACK

    dst = cv2.bitwise_not(dst)

    '''去除最大连通域，得到肺实质掩膜'''
    gray = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
    _, labels, status, _ = cv2.connectedComponentsWithStats(gray)
    parenchyma_list = list(filter(is4, np.argwhere((status > 60000) | (status < 100))))
    parenchyma_list = [parenchyma[0] for parenchyma in parenchyma_list]

    for i in range(0, len(labels)):
        for j in range(0, len(labels[i])):
            if labels[i, j] in parenchyma_list:
                dst[i, j] = BLACK

    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, element)

    '''肺实质掩膜与原图与操作得到肺实质'''
    res = cv2.bitwise_and(res, dst)

    # cv2.imshow('src', src)
    # cv2.imshow('dst', dst)
    # cv2.imshow('res', res)
    cv2.imwrite(savename, res)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    for i in range(1, 21):
        recognize_parenchyma(image_filename(0, i), image_filename(0, i, '02'))
        recognize_parenchyma(image_filename(1, i), image_filename(1, i, '02'))

    print ('Results saved as xx00002.jpg')
    # recognize_parenchyma(image_filename(0, 13), image_filename(1, 13, '02'))
