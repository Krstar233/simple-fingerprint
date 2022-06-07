import cv2 as cv
import numpy as np
import math
import Filter
from match import feature
from match import pointMatch
from test import thin
from test import direction_bw

# 增强对比度
def contrast(img_gray):
    maxlist = list()
    minlist = list()
    for line in img_gray:
        maxlist.append(max(line))
        minlist.append(min(line))
    M = max(maxlist)
    m = min(minlist)
    N = 250
    n = 50
    h, w = np.shape(img_gray)
    for i in range(h):
        for j in range(w):
            img_gray[i,j] = (N-n)*(int(img_gray[i,j])-m)/(M-m)+n
    cv.imshow('imgcontrast', img_gray)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return img_gray

# 对灰度图像进行归一化和分割处理
def normalize(img_gray):
    M0 = 150
    V0 = 2000
    Mi = np.mean(img_gray)
    Vi = np.var(img_gray)
    h, w = np.shape(img_gray)
    print(h, w)
    N = int(h/20) # 图像分割成（N*N）网格
    h1 = int(h / N) # 分割块的高度
    w1 = int(w / N) # 分割块的宽度

    for i in range(h):
        for j in range(w):
            value = img_gray[i, j]
            if value > Mi:
                x = M0 + math.sqrt((V0 * (value - Mi)**2) / Vi)
            else:
                x = M0 - math.sqrt((V0 * (value - Mi)**2) / Vi)
            img_gray[i,j] = x

    for k in range(N):
        for l in range(N):
            block = img_gray[k*h1:(k+1)*h1, l*w1:(l+1)*w1]
            # print(np.shape(block))
            M2 = np.mean(block)
            V2 = np.var(block)

            # 方差的平方小于阀值 或者 纯背景 设置成白色
            if V2 < 300 or V2 == 0:
                for i in range(h1):
                    for j in range(w1):
                        block[i, j] = 255
            else:
                Th = M2 / math.sqrt(V2)
                if Th > 60:
                    for i in range(h1):
                        for j in range(w1):
                            block[i, j] = 255

    return img_gray

def fingerTxy(filepath):
    img_filename = filepath
    img = cv.imread(img_filename)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # img_gray=cv.normalize(img_gray,dst=None,alpha=200,beta=10,norm_type=cv.NORM_MINMAX) # 对比度增强
    # sobelImage = sobelFiltering(img_gray)

    img_gray = contrast(img_gray)
    imgnormal = normalize(img_gray)  # 归一化



    binary = direction_bw(imgnormal) # 二值化1

    # binary = Filter.eightDirectionFiltering(imgnormal) # 二值化2

    clean = Filter.clear(binary)  # 去除毛刺和空洞
    # contour = Filter.thinning(binary) # 细化
    contour = thin(clean)

    cv.imshow('img', contour)
    cv.waitKey(0)
    cv.destroyAllWindows()

    h, w = np.shape(contour)
    center = (int(h / 2), int(w / 2))
    txy = feature(contour,center)
    return txy

if __name__ == '__main__':
    txy1 = fingerTxy('012_7_5.bmp')
    txy2 = fingerTxy('012_7_6.bmp')
    # txy2 = fingerTxy('022_7_6.bmp')
    result = pointMatch(txy1, txy2)  # 匹配检测
    print(result)