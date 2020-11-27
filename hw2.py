import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
from ui import Ui_MainWindow
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.onBindingUI()

    def onBindingUI(self):
        self.pushButton.clicked.connect(self.on_btn1_1_click)
        self.pushButton_2.clicked.connect(self.on_btn2_1_click)
        self.pushButton_3.clicked.connect(self.on_btn3_1_click)
        self.pushButton_4.clicked.connect(self.on_btn3_2_click)
        self.pushButton_5.clicked.connect(self.on_btn_OK_click)
        self.pushButton_6.clicked.connect(self.on_btn_cancel_click)

    def on_btn1_1_click(self):
        imgL = cv2.imread('images/imL.png',0)
        imgR = cv2.imread('images/imR.png',0)

        stereo = cv2.StereoBM_create(
	        numDisparities = 64,
	        blockSize = 9,
        )
        
        disparity = stereo.compute(imgL,imgR)

        plt.imshow(disparity,'gray')
        plt.show()


    def on_btn2_1_click(self):
        main_img = cv2.imread('images/ncc_img.jpg')
        gray_img = cv2.cvtColor(main_img,cv2.COLOR_BGR2GRAY)
        template_img = cv2.imread('images/ncc_template.jpg',0)
        height = template_img.shape[0]
        width = template_img.shape[1]
        match = cv2.matchTemplate(gray_img, template_img, cv2.TM_CCORR_NORMED)
        threshold = 0.997
        position = np.where(match >= threshold)
        for point  in zip(*position[::-1]):
            cv2.rectangle(main_img,point,(point[0]+width,point[1]+height),(0, 0, 0), 2)
        b,g,r = cv2.split(main_img)
        main_img = cv2.merge((r,g,b))
     
        plt.figure(num='matching',figsize = (16,4))
        plt.subplot(121)
        plt.imshow(match[::-1],'gray')
        plt.subplot(122)
        plt.imshow(main_img)
        plt.show()


    def on_btn3_1_click(self):
        img1 = cv2.imread('images/Aerial1.jpg')
        img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        img2 = cv2.imread('images/Aerial2.jpg')
        img2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

        SIFT = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = SIFT.detectAndCompute(img1_gray,None)
        kp2, des2 = SIFT.detectAndCompute(img2_gray,None)

        z1 = list(zip(kp1,des1))
        z2 = list(zip(kp2,des2))
        z1 = sorted(z1, key=lambda x: x[0].size,reverse=True)
        z2 = sorted(z2, key=lambda x: x[0].size,reverse=True)

        good_pt1=[]
        good_pt2=[]
        good_kp1=[]
        good_kp2=[]
        good_des1=[]
        good_des2=[]

        for i in range(20):
            # print(z1[i][0].pt)
            if(z1[i][0].pt) not in good_pt1:
                good_pt1.append(z1[i][0].pt)
                good_kp1.append(z1[i][0])
                good_des1.append(z1[i][1])
            elif(len(good_kp1)>6):
                break
        
        for i in range(20):
            # print(z2[i][0].pt)
            if(z2[i][0].pt) not in good_pt2:
                good_pt2.append(z2[i][0].pt)
                good_kp2.append(z2[i][0])
                good_des2.append(z2[i][1])
            elif(len(good_kp2)>6):
                break


        
        img_k1 = cv2.drawKeypoints(img1_gray, good_kp1[:6], img1 ,(0,255,0))
        img_k2 = cv2.drawKeypoints(img2_gray, good_kp2[:6], img2 ,(0,255,0))

        plt.figure(figsize = (16,10))
        plt.subplot(121)
        plt.imshow(img_k1)
        plt.subplot(122)
        plt.imshow(img_k2)
        plt.show()

        cv2.imwrite("images/FeatureAerial1.jpg ", img_k1)
        cv2.imwrite("images/FeatureAerial2.jpg ", img_k2)

    def on_btn3_2_click(self):
        img1 = cv2.imread('images/Aerial1.jpg')
        img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        img2 = cv2.imread('images/Aerial2.jpg')
        img2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        img_f1 = cv2.imread('images/FeatureAerial1.jpg')
        img_f2 = cv2.imread('images/FeatureAerial2.jpg')

        SIFT = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = SIFT.detectAndCompute(img1_gray,None)
        kp2, des2 = SIFT.detectAndCompute(img2_gray,None)

        z1 = list(zip(kp1,des1))
        z2 = list(zip(kp2,des2))
        z1 = sorted(z1, key=lambda x: x[0].size,reverse=True)
        z2 = sorted(z2, key=lambda x: x[0].size,reverse=True)

        # print(des1[:3])
        good_pt1=[]
        good_pt2=[]
        good_kp1=[]
        good_kp2=[]
        good_des1=[]
        good_des2=[]

        for item in z1:
            # print(item[:][0].pt)
            if(len(good_kp1)>6):
                break
            if(item[:][0].pt) not in good_pt1:
                good_pt1.append(item[:][0].pt)
                good_kp1.append(item[:][0])
                good_des1.append(item[:][1])
            
        
        for item in z2:
            # print(item[:][0].pt)
            if(len(good_kp2)>6):
                break
            if(item[:][0].pt) not in good_pt2:
                good_pt2.append(item[:][0].pt)
                good_kp2.append(item[:][0])
                good_des2.append(item[:][1])
            

        good_des1_numpy = np.asarray(good_des1, dtype=np.float32)
        good_des2_numpy = np.asarray(good_des2, dtype=np.float32)
        # print(good_des1_numpy)
        # print('')
        # print(good_des2_numpy)

        
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(good_des1_numpy,good_des2_numpy,k=2)

        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append([m])

        img_match = cv2.drawMatchesKnn(img_f1,good_kp1,img_f2,good_kp2,good,None,matchColor=(0,255,0),flags=2)
        cv2.imshow('match',img_match)
    
    def on_btn_OK_click(self):
        print("ok")
    
    def on_btn_cancel_click(self):
        print("cancel")
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

