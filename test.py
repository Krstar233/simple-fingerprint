import cv2 as cv
import numpy as np
import math
import scipy.signal as signal
from skimage import morphology
from match import feature
from match import pointMatch

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
    N = 24 # 图像分割成（N*N）网格
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

def sobelFiltering(img):
    sobelX = cv.Sobel(img, cv.CV_8U, 0, 1, 3)
    sobelY = cv.Sobel(img, cv.CV_8U, 1, 0, 3)
    cv.imshow('sobelX', sobelX)
    cv.imshow('sobelY', sobelY)
    cv.waitKey(0)
    cv.destroyAllWindows()
    sobelX = cv.Sobel(img, cv.CV_16S, 0, 1, 3)
    sobelY = cv.Sobel(img, cv.CV_16S, 1, 0, 3)
    sobel = abs(sobelX) + abs(sobelY)
    minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(sobel)
    sobelImage = cv.convertScaleAbs(sobel,cv.CV_8U, -255/maxVal, 255)
    cv.imshow('sobel', sobelImage)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return sobelImage



def filtering(im, shape=(3,3)):
    # 3*3 均值滤波
    temp = np.ones((3, 3), dtype='float')
    temp *= 1.0 / 9.0
    h, w = np.shape(im)
    In = np.zeros((h, w), dtype='uint8')
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            In[i, j] = int(im[i - 1, j - 1] * temp[0, 0] + im[i - 1, j] * temp[0, 1] + im[i - 1, j + 1] * temp[0, 2] \
                           + im[i, j - 1] * temp[1, 0] + im[i, j] * temp[1, 1] + im[i, j + 1] * temp[1, 2] \
                           + im[i + 1, j - 1] * temp[2, 0] + im[i + 1, j] * temp[2, 1] + im[i + 1, j + 1] * temp[2, 2])
    return In

def direction_bw(im):

    signal.medfilt(im, (3,3)) #中值滤波
    im = filtering(im)  # 均值滤波


    h, w = np.shape(im)

    # 判断和处理脊线方向以及切割
    Im = np.zeros((h,w),dtype='uint8')
    Icc = np.ones((h,w),dtype='uint8')
    for i in range(4, h-4):
        for j in range(4, w-4):
            sum1 = int(im[i,j-4])+im[i,j-2]+im[i,j+2]+im[i,j+4]
            sum2 = int(im[i-2,j-4])+im[i-1,j-2]+im[i+1,j+2]+im[i+2,j+4]
            sum3 = int(im[i-4,j-4])+im[i-2,j-2]+im[i+2,j+2]+im[i+4,j+4]
            sum4 = int(im[i-4,j-2])+im[i-2,j-1]+im[i+2,j+1]+im[i+4,j+2]
            sum5 = int(im[i-4,j])+im[i-2,j]+im[i+2,j]+im[i+4,j]
            sum6 = int(im[i-4,j+2])+im[i-2,j+1]+im[i+2,j-1]+im[i+4,j-2]
            sum7 = int(im[i-4,j+4])+im[i-2,j+2]+im[i+2,j-2]+im[i+4,j-4]
            sum8 = int(im[i-2,j+4])+im[i-1,j+2]+im[i+1,j-2]+im[i+2,j-4]
            sumi = [sum1,sum2,sum3,sum4,sum5,sum6,sum7,sum8]
            summax = max(sumi)
            summin = min(sumi)
            summ = math.fsum(sumi)
            b = summ / 8
            if summax+summin+4*im[i,j] > 3*b:
                sumf = summin
            else:
                sumf = summax
            if sumf > b:
                Im[i,j] = 128
            else:
                Im[i,j] = 255

    for i in range(h):
        for j in range(w):
            Icc[i,j] *= Im[i,j]

    for i in range(h):
        for j in range(w):
            if (Icc[i,j] == 128):
                Icc[i,j] = 1
            else:
                Icc[i,j] = 0


    return Icc

# 细化
def thin(binary):
    I = signal.medfilt(binary, (3,3)) # 中值滤波
    cv.imshow('I',I)
    cv.waitKey(0)
    cv.destroyAllWindows()
    h, w = np.shape(I)
    for i in range(1, h-1):
        for j in range(1, w-1):
            if I[i,j] == 0:
                if I[i,j-1]+I[i-1,j]+I[i+1,j] >= 3:
                    I[i,j] = 1
    for i in range(1, h-1):
        for j in range(1, w-1):
            if I[i,j] == 1:
                if abs(int(I[i,j+1])-int(I[i-1,j+1]))+abs(int(I[i-1,j-1])-int(I[i-1,j]))+abs(int(I[i-1,j])-int(I[i-1,j-1]))\
                        +abs(int(I[i-1,j-1])-int(I[i,j-1]))+abs(int(I[i,j-1])-int(I[i+1,j-1]))+abs(int(I[i+1,j-1])\
                        -int(I[i+1,j]))+abs(int(I[i+1,j])-int(I[i+1,j+1]))+abs(int(I[i+1,j+1])-int(I[i,j+1])) != 1:
                    if int((I[i,j+1]+I[i-1,j+1]+I[i-1,j]))*(I[i,j-1]+I[i+1,j-1]+I[i+1,j])+\
                        (I[i-1,j]+I[i-1,j-1]+I[i,j-1])*(I[i+1,j]+I[i+1,j+1]+I[i,j+1]) == 0:
                        I[i,j] = 0

    I = morphology.skeletonize(I)
    h, w = np.shape(I)
    In = np.zeros((h, w), dtype='uint8')
    for i in range(h):
        for j in range(w):
            if not I[i][j]:
                In[i][j] = 255
    for i in range(1,h-1):
        for j in range(1,w-1):
            if In[i,j] == 0:
                if In[i-1,j] == 0 and In[i,j+1] == 0 or In[i-1,j] == 0 and In[i,j-1] == 0\
                    or In[i+1,j]==0 and In[i,j-1] == 0 or In[i+1,j] == 0 and In[i,j+1] == 0:
                    In[i,j] = 255
                else:
                    In[i,j] = 0
    return In

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

def mywiener2(im, nhood=(3,3)):
    #计算局部均值
    localmean = cv.filter2D(im, -1, np.ones(nhood)) / (3*3)
    #计算局部方差
    localvar = cv.filter2D(np.multiply(im, im), -1, np.ones(nhood)) / (3*3) - localmean*localmean
    #设定整体图像的噪声方差
    noise = np.mean(localvar)

    f = im - localmean
    im = localvar - noise
    #im = max(im, 0)
    m, n = np.shape(im)
    for i in range(m):
        for j in range(n):
            if im[i, j]>0:
                im[i, j]=im[i, j]
            else:
                im[i, j]=0
    #temp = max(localvar, noise)
    temp = np.zeros(np.shape(localvar))
    m, n = np.shape(temp)
    for i in range(m):
        for j in range(n):
            if localvar[i,j] > noise:
                temp[i, j] = localvar[i, j]
            else:
                temp[i, j] = noise
    m, n = np.shape(temp)
    for i in range(m):
        for j in range(n):
            temp[i, j] = 1/temp[i, j]

    f = localmean + np.multiply(im, np.multiply(f, temp))
    return f

def centralizing(im):
    m, n = np.shape(im)
    image = mywiener2(im)
    gx, gy = np.gradient(image)
    orientnum = mywiener2(np.multiply(2*gx, gy))
    orientden = mywiener2(gx*gx - gy*gy)
    w = 8
    l1 = 9
    a=int(m/w)
    b=int(n/w)
    orient = np.zeros((a,b))
    points = int(m/w)*int(n/w)
    for i in range(points):
        x = math.floor((i)/(n/w))*w
        y = int(((i) % (m/w)) * w)
        numblock = orientnum[y:y+w-1, x:x+w-1]
        denblock = orientden[y:y+w-1, x:x+w-1]
        somma_num = sum(sum(numblock))
        somma_denom = sum(sum(denblock))
        if somma_denom!=0:
            inside = somma_num/somma_denom
            angle = 0.5*math.atan(inside)
        else:
            angle = math.pi/2

    if angle<0:
        if somma_num<0:
            angle = angle+math.pi/2
        else:
            angle = angle+math.pi
    else:
        if somma_num>0:
            angle = angle+math.pi/2

    orient[int(1+(y-1)/w), int(1+(x-1)/w)] = angle
    binarize = (orient < math.pi/2)
    bi, bj = np.nonzero(binarize)
    xdir = np.zeros((w,w))
    ydir = np.zeros((w,w))
    for k in range(len(bj)):
        i = bj[k]
        j = bi[k]
        if orient[j, i]<math.pi/2:
            x = int(l1*math.cos(orient[j,i]-math.pi/2)/(w/2))
            y = int(l1*math.sin(orient[j,i]-math.pi/2)/(w/2))
            xdir[j,i] = i-x
            ydir[j,i] = j-y
    binarize2 = np.zeros((a, b))
    for i in range(len(bj)):
        x = bj[i]
        y = bi[i]
        if not((xdir[y,x]<1) or (ydir[y,x]<1) or (xdir[y,x]>(n/w)) or (ydir[y,x]>(m/w))):
            while binarize[ydir[y,x],xdir[y,x]]>0:
                xtemp = xdir[y,x]
                ytemp = ydir[y,x]
                if xtemp<1 or ytemp<1 or xtemp>n/w or ytemp>m/w:
                    break
                x=xtemp
                y=ytemp
                if xdir[y,x]<1 or ydir[y,x]<1 or xdir[y,x]>n/w or ydir[y,x]>m/w:
                    if x-1>0:
                        while binarize[y,x-1]>0:
                            x=x-1
                            if x-1<1:
                                break
                    break
        binarize2[y, x] = binarize2[y, x]+1
    end = m/w-7
    temp = [0]*int(n/w)
    y = [0]*int(n/w)
    for j in range(n/w):
        maxone = 0
        rowindex = 0
        for k in range(end):
            if binarize2[i,j]>maxone:
                maxone=binarize2[i,j]
                rowindex=i
        temp[j]=maxone
        y[j]=rowindex
    maxone=0
    rowindex=0
    for j in range(len(temp)):
        if temp[j]>maxone:
            maxone=temp[j]
            rowindex=j
    temp2=maxone
    x=rowindex
    angle = orient[y[x],x]-math.pi/2
    xofcenter=round(x*w-(w/2)-(l1/2)*math.cos(angle))
    yofcenter=round(y[x]*w-(w/2)-(l1/2)*math.sin(angle))
    center=(xofcenter, yofcenter)
    return center

def fingerTxy(filepath):
    img_filename = filepath
    img = cv.imread(img_filename)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # img_gray=cv.normalize(img_gray,dst=None,alpha=200,beta=10,norm_type=cv.NORM_MINMAX) # 对比度增强
    # sobelImage = sobelFiltering(img_gray)

    img_gray = contrast(img_gray)
    imgnormal = normalize(img_gray)  # 归一化
    binary = direction_bw(imgnormal)  # 二值化
    clean = clear(binary)  # 去除毛刺和空洞
    contour = thin(clean)  # 细化
    cv.imshow('img', contour)
    cv.waitKey(0)
    cv.destroyAllWindows()

    h, w = np.shape(contour)
    center = (int(h / 2), int(w / 2))
    txy = feature(contour,center)
    return txy

if __name__ == '__main__' :
    txy1 = fingerTxy('finger01.png')
    txy2 = fingerTxy('finger02.png')
    result = pointMatch(txy1, txy2) #匹配检测
    print(result)