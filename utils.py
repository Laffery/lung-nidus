'''
Copyrights: ©2021 @Laffery
Date: 2021-05-24 17:16:15
LastEditor: Laffery
LastEditTime: 2021-05-31 16:04:05
'''
import numpy as np
import cv2

IMAGE_DIR = './test'
HONEY_DIR = '{}/{}'.format(IMAGE_DIR, 'honeycombing')
RETCL_DIR = '{}/{}'.format(IMAGE_DIR, 'reticular')

def image_filename(dir, index, tag='01'):
    '''
    @dir 0->honeycombing; 1->reticular\\
    @index 1~20\\
    @tag '01':src '02':肺实质 '03':病灶 '04':对比 'std': snapshot
    Return filename of target image
    '''
    DIR = RETCL_DIR if dir else HONEY_DIR
    
    if tag == 'std':
        return '{}/snapshot{}.jpg'.format(DIR, index)
    
    return '{}/{}000{}.jpg'.format(DIR, str(index).rjust(2, '0'), tag)

def image_resize(src, scale):
    '''
    @src source image\\
    @scale >1 zoom in; <1 zoom out\\
    Return resized image
    '''
    height, width = src.shape[:2]
    shape = (int(width/scale), int(height/scale))
    return cv2.resize(src, shape, interpolation=cv2.INTER_AREA)

def getConnect(src):
    '''
    @src source image\\
    Return Connected Components labels and stats
    '''
    tmp1 = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    _, tmp2 = cv2.threshold(tmp1, 128, 255, cv2.THRESH_BINARY)
    _, labels, status, _ = cv2.connectedComponentsWithStats(tmp2)
    return labels, status

def getClose(src, se):
    '''
    @src source image\\
    @se structure element shape\\
    Return close option result by se
    '''
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, se)
    return cv2.morphologyEx(src, cv2.MORPH_CLOSE, element)

def resizeSnapshot(dir, index, show=False):
    '''
    @dir 0->honeycombing; 1->reticular\\
    @index 1~20\\
    @show(False) whether to show\\
    Return filename of target image
    '''
    src = cv2.imread(image_filename(dir, index, 'std'))
    (height, width, _) = src.shape
   
    if dir:
        _width_ = 1552
        _height_ = 1552
        res = np.zeros([_height_, _width_, 3], np.uint8)
        offsetX = (int)((_width_ - width) / 2)
        offsetY = (int)((_height_ - height) / 2)

        for i in range(0, height):
            for j in range(0, width):
                res[i + offsetY, j + offsetX] = src[i, j]
        
        res = cv2.resize(res, (512, 512))

    else:
        offsetX = 28
        _width_ = width - 2 * offsetX
        offsetY = (int)((_width_ - height) / 2)
        _height_ = _width_

        res = np.zeros([_height_, _width_, 3], np.uint8)

        for i in range(0, height):
            for j in range(offsetX, width - offsetX):
                res[i + offsetY, j - offsetX] = src[i, j]

        res = cv2.resize(res, (512, 512))

    if show:
        cv2.imshow('resize ' + image_filename(dir, index, 'std'), res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return res