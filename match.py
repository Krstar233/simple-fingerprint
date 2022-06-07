import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def P(x, y, i):
    l = [(x - 1, y - 1), (x - 1, y), (x - 1, y + 1), (x, y + 1), (x + 1, y + 1), (x + 1, y), (x + 1, y - 1), (x, y - 1),
         (x - 1, y - 1)]
    return l[i - 1]


def guanghua(thin, txy, center, num):
    for j in range(num):
        txy = point(thin, center)
        pxy = list()  # pxy端点
        for v in txy:
            if v[2] == 2:
                pxy.append((v[0], v[1]))
        for i in range(len(pxy)):
            error, x, y = walk(thin, pxy[i], num)
            if error and (x, y) != (-1, -1):
                thin[x, y] = 255
    return point(thin, center)


def walk(thin, point, num):
    x0, y0 = point
    x1, y1 = (-1, -1)
    error = False
    thin[x0, y0] = 255
    for i in range(num):
        if error:
            return error, x1, y1
        t1 = 0
        for j in range(1, 9):
            x, y = P(x0, y0, j)
            v = 0 if thin[x, y] == 255 else 1
            if v and (x, y) != (x1, y1):
                x1, y1 = x0, y0
                x0, y0 = x, y
                break
        for j in range(1, 9):
            x, y = P(x0, y0, j)
            v = 0 if thin[x, y] == 255 else 1
            t1 += v
        if t1 == 0 or t1 > 2:
            error = True
            return error, x1, y1
    return error, x1, y1


def removeTooClose(txy, distance):
    pxy = list()
    bxy = list()
    for i in range(len(txy)):
        if txy[i][2] == 2:
            pxy.append((txy[i][0], txy[i][1]))
        else:
            bxy.append((txy[i][0], txy[i][1]))

    for i in range(len(pxy)):
        for j in range(len(pxy)):
            if pxy[i] == pxy[j]:
                continue
            d = np.sqrt((pxy[i][0] - pxy[j][0]) ** 2 + (pxy[i][1] - pxy[j][1]) ** 2)
            if d < distance:
                pxy[j] = (-1, -1)

    for i in range(len(bxy)):
        for j in range(len(bxy)):
            if bxy[i] == bxy[j]:
                continue
            d = np.sqrt((bxy[i][0] - bxy[j][0]) ** 2 + (bxy[i][1] - bxy[j][1]) ** 2)
            if d < distance:
                bxy[j] = (-1, -1)

    txy = list()
    for x, y in pxy:
        if x == -1 or y == -1:
            continue
        txy.append([x, y, 2])
    for x, y in bxy:
        if x == -1 or y == -1:
            continue
        txy.append([x, y, 6])
    return txy

def point(img, center):
    txy = list()
    x, y = center
    h, w = np.shape(img)
    b = int(min(h, w) / 4)
    for i in range(x - b, x + b):
        for j in range(y - b, y + b):
            if img[i][j] == 0:
                cn = 0
                for k in range(1, 9):
                    x1, y1 = P(i, j, k)
                    x2, y2 = P(i, j, k + 1)
                    v1 = 0 if img[x1, y1] == 255 else 1
                    v2 = 0 if img[x2, y2] == 255 else 1
                    cn += np.abs(v1 - v2)
                if cn == 2:
                    txy.append([i, j, 2])
                elif cn == 6:
                    txy.append([i, j, 6])
    return txy

# center 中心点坐标
def feature(img, center):
    # flag --- 是否去掉过于近的点
    h ,w = np.shape(img)
    txy = point(img, center)
    txy = guanghua(img, txy, center, int(max(h,w)/80))
    txy = removeTooClose(txy, int(max(h,w)/30))

    In = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    for v in txy:
        if v[2] == 2:
            In[v[0],v[1]] = [255, 0, 0]
        elif v[2] == 6:
            In[v[0], v[1]] = [0, 255, 0]

    return txy

def pointMatch(txy1, txy2):
    p, q = getBasicPoint(txy1, txy2)
    if p[0] == -1:
        print('完全匹配失败')
        return False
    n = len(txy1)
    m = len(txy2)
    minnm = min(n, m)
    num = 20 if 20 < minnm else minnm - 1
    p_s = findPoint(p[0], p[1], txy1, num)
    q_s = findPoint(q[0], q[1], txy2, num)

    count1 = 0
    count2 = 0
    for p in p_s:
        if p[2] == 2:
            count1 += 1
    for q in q_s:
        if q[2] == 2:
            count2 += 1
    delta = np.abs(count1-count2)
    T = num / 10 # 阈值
    print('匹配率：{}%'.format((1-delta / num) * 100))
    if delta < T:
        print('匹配成功')
        return True
    print('匹配失败')
    return False

def getBasicPoint(txy1, txy2):
    for p_i in txy1:
        ps = findPoint(p_i[0], p_i[1],txy1, 2)
        p1 = ps[0]
        p2 = ps[1]
        thetas1 = getThetas((p1[0], p1[1]), (p2[0], p2[1]), (p_i[0], p_i[1]))
        for q_i in txy2:
            if q_i[2] != p_i[2]:
                continue
            qs = findPoint(q_i[0], q_i[1], txy2, 2)
            q1 = qs[0]
            q2 = qs[1]
            thetas2 = getThetas((q1[0],q1[1]), (q2[0],q2[1]), (q_i[0],q_i[1]))
            delta = 0
            b = 0.2 # 误差阈值（越小越精确)
            for i in range(3):
                delta += np.abs(thetas1[i]-thetas2[i])
            if delta < b:
                if p1[2] == q1[2] and p2[2] == q2[2] or p1[2] == q2[2] and p2[2] == q1[2]:
                    return p_i, q_i
                continue
    return [-1,-1, 0], [-1, -1, 0]

def getThetas(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    a = np.sqrt((x1-x2)**2+(y1-y2)**2)
    b = np.sqrt((x1-x3)**2+(y1-y3)**2)
    c = np.sqrt((x2-x3)**2+(y2-y3)**2)

    angleA = np.arccos((b * b + c * c - a * a) / (2 * b * c))
    angleB = np.arccos((a * a + c * c - b * b) / (2 * a * c))
    angleC = np.arccos((a * a + b * b - c * c) / (2 * a * b))

    thetas = [angleA, angleB, angleC]
    thetas.sort()
    return thetas

def findPoint(x0, y0, txy, num):
    if len(txy) <= num:
        return None
    p_s = list()
    dis_s = list()
    ntxy = list()
    for p in txy:
        x1, y1 = p[0], p[1]
        dis = np.sqrt((x1-x0)**2+(y1-y0)**2)
        if dis != 0:
            dis_s.append(dis)
            ntxy.append(p)

    for i in range(num):
        min_d = min(dis_s)
        index = dis_s.index(min_d)
        p_s.append(ntxy[index])
        del ntxy[index]
        del dis_s[index]
    return p_s

if __name__ == '__main__':
    img = cv.imread('finger.png')
    plt.imshow(img)
    plt.show()