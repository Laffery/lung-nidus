import numpy as np
import cv2

IMAGE_DIR = './test'
HONEY_DIR = '{}/{}'.format(IMAGE_DIR, 'honeycombing')
RETCL_DIR = '{}/{}'.format(IMAGE_DIR, 'reticular')

def image_filename(dir, index, tag='01'):
    '''
    @dir 0->honeycombing; 1->reticular\\
    @index 1~20\\
    @tag '01':src '02':肺实质 '03':病灶 
    Return filename of target image
    '''
    DIR = RETCL_DIR if dir else HONEY_DIR
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
