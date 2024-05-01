import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from cvzone.ClassificationModule import Classifier
import math
cap = cv2.VideoCapture(0)
detector=HandDetector(maxHands=1)
classifier=Classifier("keras_model.h5","Model/labels.txt")
counter=0
folder="Data/A"
labels=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
while True:
    success, img = cap.read()
    imgOutput=img.copy()
    hands,img=detector.findHands(img)
    if hands:
        hand=hands[0]
        x,y,w,h=hand['bbox']
        imgwhite = np.ones((300,300, 3), np.uint8) * 255
        imgCrop=img[y-20:y+h+20,x-20:x+w+20]
        imgCropShape=imgCrop.shape
        asp=h/w
        if asp>1:
            c=300/h
            cal=math.ceil(c*w)
            imgReSize=cv2.resize(imgCrop,(min(cal,300),300))
            imgReSizeShape=imgReSize.shape
            gap=math.ceil((300-cal)/2)
            imgwhite[:,gap:cal+gap] = imgReSize
            prediction,index=classifier.getPrediction(imgwhite)
            print(prediction,index)
        else:
            c=300/h
            hcal=math.ceil(c*h)
            imgReSize=cv2.resize(imgCrop,(min(hcal,300),300))
            imgReSizeShape=imgReSize.shape
            hgap=math.ceil((300-hcal)/2)
            imgwhite[:,hgap:hcal+hgap]=imgReSize
            prediction, index = classifier.getPrediction(imgwhite)
            print(prediction, index)
        cv2.putText(imgOutput,labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
        cv2.imshow("ImageWhite",imgwhite)
    cv2.imshow("Image", img)
    cv2.waitKey(1)