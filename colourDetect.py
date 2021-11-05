import cv2
import numpy as np

imgPath = 'src/1.jpg'
img = cv2.imread(imgPath)
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

def empty(e):
    pass

if __name__ == '__main__': 
    cv2.namedWindow('TrackBars')        # 创建拖动条界面
    cv2.resizeWindow('TrackBars', 640, 330)
    cv2.createTrackbar('Hue Min', 'TrackBars', 64, 179, empty)
    cv2.createTrackbar('Hue Max', 'TrackBars', 84, 179, empty)
    cv2.createTrackbar('Sat Min', 'TrackBars', 56, 255, empty)
    cv2.createTrackbar('Sat Max', 'TrackBars', 155, 255, empty)
    cv2.createTrackbar('Val Min', 'TrackBars', 84, 255, empty)
    cv2.createTrackbar('Val Max', 'TrackBars', 162, 255, empty)

    imgHSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)   # 转换为HSV模型

    infoBefore = []
    while True:
        h_min = cv2.getTrackbarPos('Hue Min', 'TrackBars')
        h_max = cv2.getTrackbarPos('Hue Max', 'TrackBars')
        s_min = cv2.getTrackbarPos('Sat Min', 'TrackBars')
        s_max = cv2.getTrackbarPos('Sat Max', 'TrackBars')
        v_min = cv2.getTrackbarPos('Val Min', 'TrackBars')
        v_max = cv2.getTrackbarPos('Val Max', 'TrackBars')
        info = [h_min, h_max, s_min, s_max, v_min, v_max]
        if info != infoBefore:
            print(info)
            infoBefore = info
        maskL = np.array([h_min, s_min, v_min])
        maskH = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(imgHSV, maskL, maskH)                # 设置介于L和H之间的蒙版
        imgResult = cv2.bitwise_and(img, img, mask=mask)        # 与蒙版与，提取区域图形

        imgStack = stackImages(0.5, ([img, imgHSV], [mask, imgBlank]))        # 拼合图像
        cv2.imshow('result', imgStack)
        cv2.waitKey(1)
