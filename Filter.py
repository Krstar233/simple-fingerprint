import numpy as np
import cv2 as cv
from test import normalize
from test import thin

def eightDirectionFiltering(img):
    mat1 = [[-10, -20, -30, -30, -30, -20, -10],
            [2, 4, 6, 6, 6, 4, 2],
            [4, 8, 12, 12, 12, 8, 4],
            [8, 16, 24, 24, 24, 16, 8],
            [4, 8, 12, 12, 12, 8, 4],
            [2, 4, 6, 6, 6, 4, 2],
            [-10, -20, -30, -30, -30, -20, -10]]

    mat2 = [[4, 2, 4, -10, -20, -30, -30],
            [8, 8, 12, 6, 6, -30, -20],
            [4, 16, 24, 12, 6, 4, -10],
            [2, 8, 12, 24, 12, 8, 2],
            [-10, 4, 6, 12, 24, 16, 4],
            [-20, -30, 6, 6, 12, 8, 8],
            [-30, -30, -20, -10, 4, 2, 4]]

    mat3 = [[8, 4, 2, -10, -20, -30, -30],
            [4, 16, 8, 4, 6, 6, -30],
            [2, 8, 24, 12, 12, 6, -20],
            [-10, 4, 12, 24, 12, 4, -10],
            [-20, 6, 12, 12, 24, 8, 2],
            [-30, 6, 6, 4, 8, 16, 4],
            [-30, -30, -20, -10, 2, 4, 8]]

    mat4 = [[4, 8, 4, 2, -10, -20, -30],
            [2, 8, 16, 8, 4, -30, -30],
            [4, 12, 24, 12, 6, 6, -20],
            [-10, 6, 12, 24, 12, 6, -10],
            [-20, 6, 6, 12, 24, 12, 4],
            [-30, -30, 4, 8, 16, 8, 2],
            [-30, -20, -10, 2, 4, 8, 4]]

    mat5 = [[-10, 2, 4, 8, 4, 2, -10],
            [-20, 4, 8, 16, 8, 4, -20],
            [-30, 6, 12, 24, 12, 6, -30],
            [-30, 6, 12, 24, 12, 6, -30],
            [-30, 6, 12, 24, 12, 6, -30],
            [-20, 4, 8, 16, 8, 4, -20],
            [-10, 2, 4, 8, 4, 2, -10]]

    mat6 = [[-30, -20, -10, 2, 4, 8, 4],
            [-30, -30, 4, 8, 16, 8, 2],
            [-20, 6, 6, 12, 24, 12, 4],
            [-10, 6, 12, 24, 12, 6, -10],
            [4, 12, 24, 12, 6, 6, -20],
            [2, 8, 16, 8, 4, -30, -30],
            [4, 8, 4, 2, -10, -20, -30]]

    mat7 = [[-30, -30, -20, -10, 2, 4, 8],
            [-30, 6, 6, 4, 8, 16, 4],
            [-20, 6, 12, 12, 24, 8, 2],
            [-10, 4, 12, 24, 12, 4, -10],
            [2, 8, 24, 12, 12, 6, -20],
            [4, 16, 8, 4, 6, 6, -30],
            [8, 4, 2, -10, -20, -30, -30]]

    mat8 = [[-30, -30, -20, -10, 4, 2, 4],
            [-20, -30, 6, 6, 12, 8, 8],
            [-10, 4, 6, 12, 24, 16, 4],
            [2, 8, 12, 24, 12, 8, 2],
            [4, 16, 24, 12, 6, 4, -10],
            [8, 8, 12, 6, 6, -30, -20],
            [4, 2, 4, -10, -20, -30, -30]]

    mat = [mat1, mat2, mat3, mat4, mat5, mat6, mat7, mat8]
    mat = np.array(mat, dtype='float')

    results = list()
    for m in mat:
        result = cv.filter2D(img,3, m)
        results.append(result)

    print('Synthesizing...')
    img_dir = np.zeros(np.shape(img), dtype='uint8')
    h, w = np.shape(img)
    b = int(h / 20)
    for i in range(b, h - b):
        for j in range(b, w - b):
            M = list()
            V = list()
            for k in range(len(results)):
                m_k = np.mean(results[k][i-b:i+b, j-b:j+b])
                v_k = np.var(results[k][i-b:i+b, j-b:j+b])
                M.append(m_k)
                V.append(v_k)
            vmax = max(V)
            index = V.index(vmax)
            H_en = results[index][i, j] - M[index]
            p = 100
            q = 0
            H_nen = np.arctan(p * (H_en - q)) / (np.pi / 2) * 256
            img_dir[i, j] = H_nen
            if img_dir[i, j] < 128:
                img_dir[i, j] = 0
            else:
                img_dir[i, j] = 1

    cv.imshow('img_dir', img_dir)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return img_dir

def clear(im):
    # 去除指纹中的空洞和毛刺
    m, n = np.shape(im)
    Icc = im
    for i in range(m):
        for j in range(n):
            if Icc[i, j] == 1:
                Icc[i, j] = 0
            else:
                Icc[i, j] = 1

    # 去除毛刺
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            if Icc[i, j] == 0:
                if Icc[i - 1, j] + Icc[i + 1, j] + Icc[i, j - 1] + Icc[i, j + 1] >= 3:
                    Icc[i, j] = 1
                else:
                    Icc[i, j] = Icc[i, j]
    # 去除空洞
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            if Icc[i, j] == 1:
                if abs(int(Icc[i - 1, j - 1]) - int(Icc[i - 1, j])) + abs(int(Icc[i - 1, j]) - int(Icc[i - 1, j + 1])) + \
                        abs(int(Icc[i - 1, j + 1]) - int(Icc[i, j + 1])) + abs(int(Icc[i, j + 1]) - int(Icc[i + 1, j + 1])) \
                        + abs(int(Icc[i + 1, j + 1]) - int(Icc[i + 1, j])) + abs(int(Icc[i + 1, j]) - int(Icc[i + 1, j - 1])) + \
                        abs(int(Icc[i + 1, j - 1]) - int(Icc[i, j - 1])) + abs(int(Icc[i, j - 1]) - int(Icc[i - 1, j - 1])) != 1:
                    if (Icc[i - 1, j - 1] + Icc[i, j - 1] + Icc[i - 1, j]) * (
                            Icc[i + 1, j + 1] + Icc[i + 1, j] + Icc[i, j + 1]) + (
                            Icc[i - 1, j + 1] + Icc[i - 1, j] + Icc[i, j + 1]) \
                            * (Icc[i + 1, j - 1] + Icc[i, j - 1] + Icc[i + 1, j]) == 0:
                        Icc[i, j] = 0

    for i in range(m):
        for j in range(n):
            if Icc[i, j] == 0:
                Icc[i, j] = 1
            else:
                Icc[i, j] = 0
    Icc = thinning(Icc)
    return Icc

def thinning(im):
    # 用开运算和闭运算对图像进行细化操作
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))  # 定义用于做开闭运算的结构元素
    # 先腐蚀后膨胀做开运算
    im = cv.morphologyEx(im, cv.MORPH_OPEN, kernel)
    # 先膨胀后腐蚀做闭运算
    im = cv.morphologyEx(im, cv.MORPH_CLOSE, kernel)
    return im

if __name__ == '__main__':
    img_filename = 'finger2.bmp'
    img = cv.imread(img_filename)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    normalize(img_gray)
    img_min = cv.blur(img_gray, (5,5))

    cv.imshow('img_blur',img_min)
    cv.waitKey(0)
    cv.destroyAllWindows()

    img_dir = eightDirectionFiltering(img_min)
    cv.imshow('imgDir', img_dir)

    cv.waitKey(0)
    cv.destroyAllWindows()

    clean = clear(img_dir)
    thinned = thin(clean)