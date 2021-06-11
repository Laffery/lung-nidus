'''
Copyrights: ©2021 @Laffery
Date: 2021-05-28 08:31:04
LastEditor: Laffery
LastEditTime: 2021-06-11 11:38:48
'''
# coding: utf-8

import numpy as np
from skimage import data
import matplotlib.pyplot as plt
import fast_glcm
from PIL import Image
from utils import image_filename, getConnect, getClose, resizeSnapshot
import cv2

def YELLOW(bgr):
    return bgr[1] == bgr[2] and bgr[0] < bgr[1]

def GREEN(bgr):
    return bgr[0] == bgr[2] and bgr[0] < bgr[1]

def PURPLE(bgr):
    return bgr[0] == bgr[2] and bgr[0] > bgr[1]

def distinguishNidus(dir, index, show=False, se=(20, 20)):
    src = cv2.imread(image_filename(dir, index, '03'))
    res = cv2.imread(image_filename(dir, index, '01'))
    (totalHeight, totalWidth, _) = src.shape


    # 副本进行闭操作得到大概连通的图
    copy = getClose(src, se)

    # 计算副本及原图的连通域
    labeltmp, statustmp = getConnect(copy)
    label, status = getConnect(src)

    # 遍历副本连通域，计算每个连通域里原图中大颗粒所占比例
    for connectNum in range(1, len(statustmp)):
        totalSize = 0
        totalConnect = 0
        particles = []

        # 对每个连通域：
        for i in range(0, totalHeight):
            for j in range(0, totalWidth):
                if labeltmp[i, j] == connectNum:
                    if label[i, j] and label[i, j] not in particles:
                        particles.append(label[i, j])
                        totalConnect += status[label[i, j], 4]
                    totalSize = totalSize + 1

        if totalSize == 0:
            continue

        rat = totalConnect / totalSize

        for i in range(0, totalHeight):
            for j in range(0, totalWidth):
                if labeltmp[i, j] == connectNum:
                    res[i, j] = [128, 0, 128] if rat > 0.4 else [0, 128, 0]
    
    if show:
        # result show time
        cv2.imshow('distinguish result', res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return res

def calcIoU(dir, index, expe, ctrl, save=False):
    '''
    计算交并比 \\
    @dir: 目录 \\
    @index: 图片索引 \\
    @expe: 实验组识别函数 \\
    @ctrl: 对照组识别函数 \\
    @save: 是否将生成的对比图保存 \\
    '''
    res = distinguishNidus(dir, index)
    std = resizeSnapshot(dir, index)

    miss = 0
    hit = 0

    for i in range(100, 420):
        for j in range(100, 420):
            if ctrl(std[i, j]):
                if expe(res[i, j]):
                    hit = hit + 1
                else:
                    miss = miss + 1

    if save:
        view = np.zeros([512, 1024, 3], np.uint8)
        for i in range(0, 512):
            for j in range(0, 512):
                view[i, j] = std[i, j]
                view[i, j + 512] = res[i, j]

        cv2.imwrite(image_filename(dir, index, '04'), view)

    return hit/(miss + hit) if miss + hit else 0

def main():
    '''
    实验组网状为绿色，蜂窝为紫色 \\
    对照组：\\
    蜂窝目录下，蜂窝为紫色，网状为黄色 \\
    网状目录下，没有蜂窝状，网状为黄色 \\
    '''
    for index in range(1, 21):
        iou = calcIoU(0, index, PURPLE, PURPLE, True)
        print(f'{iou} #蜂窝 {index}')
        iou = calcIoU(0, index, GREEN, YELLOW)
        print(f'{iou} #蜂窝之网状 {index}')
        iou = calcIoU(1, index, GREEN, YELLOW, True)
        print(f'{iou} #网状 {index}')
        iou = calcIoU(1, index, PURPLE, PURPLE)
        print(f'{iou} #网状之蜂窝 {index}')

if __name__ == '__main__':
    main()

# def GLCM():
    '''
    GLCM尝试，效果不佳
    '''
    # image = image_filename(0, 2, '03')
    # img = np.array(Image.open(image).convert('L'))
    # h,w = img.shape

    # mean = fast_glcm.fast_glcm_mean(img)
    # std = fast_glcm.fast_glcm_std(img)
    # cont = fast_glcm.fast_glcm_contrast(img)
    # diss = fast_glcm.fast_glcm_dissimilarity(img)
    # homo = fast_glcm.fast_glcm_homogeneity(img)
    # asm, ene = fast_glcm.fast_glcm_ASM(img)
    # ma = fast_glcm.fast_glcm_max(img)
    # ent = fast_glcm.fast_glcm_entropy(img)

    # plt.figure(figsize=(10,4.5))
    # fs = 10
    # plt.subplot(2,5,1)
    # plt.tick_params(labelbottom=False, labelleft=False)
    # plt.imshow(img)
    # plt.title('original', fontsize=fs)

    # plt.subplot(2,5,2)
    # plt.tick_params(labelbottom=False, labelleft=False)
    # plt.imshow(mean)
    # plt.title('mean', fontsize=fs)

    # plt.subplot(2,5,3)
    # plt.tick_params(labelbottom=False, labelleft=False)
    # plt.imshow(std)
    # plt.title('std', fontsize=fs)

    # plt.subplot(2,5,4)
    # plt.tick_params(labelbottom=False, labelleft=False)
    # plt.imshow(cont)
    # plt.title('contrast', fontsize=fs)

    # plt.subplot(2,5,5)
    # plt.tick_params(labelbottom=False, labelleft=False)
    # plt.imshow(diss)
    # plt.title('dissimilarity', fontsize=fs)

    # plt.subplot(2,5,6)
    # plt.tick_params(labelbottom=False, labelleft=False)
    # plt.imshow(homo)
    # plt.title('homogeneity', fontsize=fs)

    # plt.subplot(2,5,7)
    # plt.tick_params(labelbottom=False, labelleft=False)
    # plt.imshow(asm)
    # plt.title('ASM', fontsize=fs)

    # plt.subplot(2,5,8)
    # plt.tick_params(labelbottom=False, labelleft=False)
    # plt.imshow(ene)
    # plt.title('energy', fontsize=fs)

    # plt.subplot(2,5,9)
    # plt.tick_params(labelbottom=False, labelleft=False)
    # plt.imshow(ma)
    # plt.title('max', fontsize=fs)

    # plt.subplot(2,5,10)
    # plt.tick_params(labelbottom=False, labelleft=False)
    # plt.imshow(ent)
    # plt.title('entropy', fontsize=fs)

    # plt.tight_layout(pad=0.5)
    # plt.show()