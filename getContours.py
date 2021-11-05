import cv2
import numpy as np
from numpy.core.numeric import count_nonzero

imgPath = 'src/shapes.png'
img = cv2.imread(imgPath)
imgCnt = img.copy()

imgWid = img.shape[1]
imgHei = img.shape[2]
imgBlank = np.zeros((imgHei, imgWid, 3), np.uint8)

def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def getContours(cimg):
    contours, hierarchy = cv2.findContours(cimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 寻找轮廓
    for cnt in contours: 
        area = cv2.contourArea(cnt)     # 计算轮廓面积
        print(area)
        if area > 500:      # 轮廓面积大于一定值，不是噪点
            cv2.drawContours(imgCnt, cnt, -1, (255, 0, 0), 3)   # 画轮廓
            peri = cv2.arcLength(cnt, True)             # 计算周长
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)   # 找拐角
            objCorner = len(approx)
            if objCorner == 3: 
                objText = 'Triangle'
            elif objCorner == 4:
                objText = 'Rectangle'
            else: 
                objText = str(objCorner) + ' corners'
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgCnt, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(imgCnt, objText, (x, y+h//2), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)

if __name__ == '__main__': 
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (17, 7), 1)
    imgCanny = cv2.Canny(imgBlur, 50, 50)
    getContours(imgCanny)

    imgStack = stackImages(0.7, ([img, imgGray, imgBlur], [imgCanny, imgCnt, imgBlank]))        # 拼合图像
    cv2.imshow('result', imgStack)
    cv2.waitKey(0)
