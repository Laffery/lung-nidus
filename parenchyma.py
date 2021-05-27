from utils import image_filename, image_savename
import cv2
import numpy as np

WHITE = [255, 255, 255]
BLACK = [0, 0, 0]

def recognize_parenchyma(filename):
    '''
    分割图片中的肺实质（lung parenchyma）\\
    filename 为图片文件名
    '''
    src = cv2.imread(filename)
    width = src.shape[1]
    height = src.shape[0]
    # cv2.imshow('src', src)

    res = src
    scale = 5
    offset = (int)(height / scale)

    for raw in range(height):
        if raw <= offset or raw >= (scale - 1) * offset:
            res[raw] = [ BLACK ] * width

    # cv2.imshow('res', res)

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, src = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    _, labels, status, _ = cv2.connectedComponentsWithStats(src)

    def is4(n):
        return n[1] == 4

    trachea_list = list(filter(is4, np.argwhere((status <= 2000) & (status >= 400))))
    trachea_list = [trachea[0] for trachea in trachea_list]

    if len(trachea_list) >= 0:
        for i in range(0, height):
            for j in range(0, width):
                if(labels[i, j] in trachea_list):
                    src[i, j] = 0

    src = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    dst = cv2.morphologyEx(src, cv2.MORPH_CLOSE, element)

    gray = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
    _, labels, status, _ = cv2.connectedComponentsWithStats(gray)

    print (status)

    parenchyma_list = list(filter(is4, np.argwhere(status >= 60000)))
    parenchyma_list = [parenchyma[0] for parenchyma in parenchyma_list]

    for i in range(0, len(labels)):
        for j in range(0, len(labels[i])):
            if labels[i, j] in parenchyma_list:
                res[i, j] = BLACK
                # pass


    # cv2.imshow('src', src)
    cv2.imshow('dst', dst)
    cv2.imshow('res', res)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def recognize_parenchyma1(filename, savename):
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

    for raw in range(height):
        if raw <= offset or raw >= (scale - 1) * offset:
            res[raw] = [ BLACK ] * width

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, src = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    _, labels, status, _ = cv2.connectedComponentsWithStats(src)

    def is4(n):
        return n[1] == 4

    trachea_list = list(filter(is4, np.argwhere((status <= 2000) & (status >= 400))))
    trachea_list = [trachea[0] for trachea in trachea_list]

    if len(trachea_list):
        for i in range(0, height):
            for j in range(0, width):
                if(labels[i, j] in trachea_list):
                    src[i, j] = 0

    src = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    dst = cv2.morphologyEx(src, cv2.MORPH_CLOSE, element)
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


    # gray = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
    # _, labels, status, _ = cv2.connectedComponentsWithStats(gray)
    # parenchyma_list = list(filter(is4, np.argwhere((status > 60000) | (status < 100))))
    # parenchyma_list = [parenchyma[0] for parenchyma in parenchyma_list]

    # for i in range(0, len(labels)):
    #     for j in range(0, len(labels[i])):
    #         if labels[i, j] in parenchyma_list:
    #             dst[i, j] = BLACK

    # res = cv2.bitwise_and(res, dst)

    # cv2.imshow('src', src)
    cv2.imshow('dst', dst)
    # cv2.imshow('res', res)
    # cv2.imwrite(savename, res)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # for i in range(1, 21):
    #     recognize_parenchyma1(image_filename(0, i), image_savename(0, i))
    #     recognize_parenchyma1(image_filename(1, i), image_savename(1, i))

    # print ('Results saved as xx00002.jpg')
    recognize_parenchyma1(image_filename(1, 13), image_savename(1, 13))
