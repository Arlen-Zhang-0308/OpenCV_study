import cv2
import numpy as np

cascadeXml = 'haarcascade_frontalface_default.xml'
plateCascade = cv2.CascadeClassifier(cascadeXml)    # 导入分类特征
imgPath = 'src/face1.jpg'
img = cv2.imread(imgPath)

imgWid = 480                                        # 修改后图像宽度
imgHei = imgWid * img.shape[0]//img.shape[1]        # 成比例
img = cv2.resize(img, (imgWid, imgHei))
imgCnt = img.copy()                                 # 画轮廓的图像

imgBlank = np.zeros((imgHei, imgWid, 3), np.uint8)  # 空白（黑）图像

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


if __name__ == '__main__': 
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plates = plateCascade.detectMultiScale(imgGray)

    for x, y, w, h in plates: 
        cv2.rectangle(imgCnt, (x, y), (x+w, y+h), (0, 0, 255), 2)
    imgResult = stackImages(1, [img, imgGray, imgCnt])  # 拼合图像
    cv2.imshow('result', imgResult)
    cv2.waitKey(0)