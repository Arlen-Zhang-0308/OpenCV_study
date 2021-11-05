import cv2
import numpy as np

imgPath = 'src/paper.jpg'
img = cv2.imread(imgPath)

imgWid = 480                                        # 修改后图像宽度
imgHei = imgWid * img.shape[0]//img.shape[1]        # 成比例
img = cv2.resize(img, (imgWid, imgHei))
imgCnt = img.copy()                                 # 画轮廓的图像

imgBlank = np.zeros((imgHei, imgWid, 3), np.uint8)  # 空白（黑）图像

# imgWid = 640
# imgHei = 480
# cam = cv2.VideoCapture(1)
# cam.set(10, 150)
# suc, img = cam.read()
# img = cv2.resize(img, (imgWid,imgHei))
# imgCnt = img.copy()
# imgBlank = np.zeros((imgHei, imgWid, 3), np.uint8)  # 空白（黑）图像

''' 拼合图像 '''
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

''' 图像预处理 '''
def preProcessing(pimg): 
    imgGray = cv2.cvtColor(pimg, cv2.COLOR_BGR2GRAY)        # 灰度图
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)          # 高斯模糊
    imgCanny = cv2.Canny(imgBlur, 200, 200)                 # 获取轮廓
    kern = np.ones((5, 5))      # 卷积核
    imgDial = cv2.dilate(imgCanny, kern, iterations=2)      # 膨胀
    imgThreshold = cv2.erode(imgDial, kern, iterations=2)   # 腐蚀

    return imgThreshold

''' 给ndarray的指定维度赋值 '''
def npAssign(array, index, dat, axis=0): 
    narray = np.delete(array, index, axis=axis)
    return np.insert(narray, index, dat, axis=axis)

''' 获取轮廓 '''
def getContours(cimg, cCnt):
    contours, hierarchy = cv2.findContours(cimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 寻找轮廓
    maxArea = 0
    biggestVertex = 0
    for cnt in contours: 
        area = cv2.contourArea(cnt)     # 计算轮廓面积
        if area > 5000:      # 轮廓面积大于一定值，不是噪点
            cv2.drawContours(cCnt, cnt, -1, (255, 0, 0), 3)   # 画轮廓
            peri = cv2.arcLength(cnt, True)             # 计算周长
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)   # 找拐角
            objCorner = len(approx)                     # 拐角（顶点）个数
            if area > maxArea and objCorner == 4:       # 判断是矩形的纸张
                biggestVertex = approx
                maxArea = area
                cv2.drawContours(cCnt, biggestVertex, -1, (0, 255, 0), 20)    # 画顶点
        
    return biggestVertex

''' 将图片的部分变换为矩形 '''
def getWarp(wimg, vertexCor): 
    try: 
        if vertexCor == 0:      # 如果没有通过getContours获取到矩形轮廓顶点， 则直接返回空白图像
            return imgBlank
    except: 
        pass
    dis = []
    for cord in vertexCor:      # 将每个顶点的x, y坐标相加，来判断距离原点的距离
        dis.append(cord[0][0]+cord[0][1])
    for i in range(len(dis)):   # 按照dis的距离，对dis和vertexCor中的值进行递增排序（选择排序）
        smallest = i
        for j in range(i, len(dis)): 
            if dis[smallest] > dis[j]: 
                smallest = j
        if smallest != i: 
            a = dis[smallest]
            dis[smallest] = dis[i]
            dis[i] = a
            b = vertexCor[smallest, ...]
            vertexCor = npAssign(vertexCor, smallest, vertexCor[i, ...])
            vertexCor = npAssign(vertexCor, i, b)

    if vertexCor[1][0][0] < vertexCor[2][0][0]:     # 若dis的对应值相等，第二个点必定是横坐标大于第三个点的
        a = vertexCor[1, ...]
        vertexCor = npAssign(vertexCor, 1, vertexCor[2, ...])
        vertexCor = npAssign(vertexCor, 2, a)
        
    cord1 = np.float32(vertexCor)
    cord2 = np.float32([[0, 0], [imgWid, 0], [0, imgHei], [imgWid, imgHei]])
    matrix = cv2.getPerspectiveTransform(cord1, cord2)              # 创建透视变换矩阵
    WarpOut = cv2.warpPerspective(wimg, matrix, (imgWid, imgHei))   # 依照矩阵进行变换
    return WarpOut

''' 主函数 '''
if __name__ == '__main__': 
    # suc, img = cam.read()                       # 获得摄像头图像
    # img = cv2.resize(img, (imgWid,imgHei))
    # imgCnt = img.copy()

    imgThreshold = preProcessing(img)           # 预处理获取轮廓
    vertexCord = getContours(imgThreshold, imgCnt)      # 获取轮廓顶点
    imgWarp = getWarp(img, vertexCord)          # 将顶点表示四边形变换为矩形

    imgResult = stackImages(0.7, [img, imgThreshold, imgCnt, imgWarp])  # 拼合图像
    cv2.imshow('result', imgResult)
    cv2.waitKey(0)
