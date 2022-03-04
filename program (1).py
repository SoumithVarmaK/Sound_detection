import pygame
import cv2
import numpy as np
import pygame
from pygame import mixer
from playsound import playsound
import os

cap = cv2.VideoCapture(0)
wht = 320
confThreshold = 0.5
nmsThreshold = 0.3

classesFile = 'coco.names'
classNames = []
with open(classesFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
modelConfi = 'yolov3.cfg'
modelwei = 'yolov3.weights'
net = cv2.dnn.readNetFromDarknet(modelConfi,modelwei)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findobj(outputs,img):
    ht,wt,ct= img.shape
    bbox=[]
    classIds=[]
    confs=[]

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence =scores[classId]
            if confidence > confThreshold:
                w,h=int(det[2]*wt),int(det[3]*ht)
                x,y = int((det[0]*wt)-w/2),int((det[1]*ht)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)
    #print(indices)
    for i in indices:
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),2)
        cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),3)
        x=classNames[classIds[i]]
        if x=="horse" or x=="elephant" or x=="zebra" or x=="giraffe":
            os.system('C:\\Users\\jeswa\\PycharmProjects\\Scarecrow\\sounds\\CrackersSound.mp3')
        elif x=="cat" or x=="dog" or x=="sheep":
            os.system('C:\\Users\\jeswa\\PycharmProjects\\Scarecrow\\sounds\\LionRoarSound.mp3')
        elif x=="bird":
            os.system('C:\\Users\\jeswa\\PycharmProjects\\Scarecrow\\sounds\\OwlSound.mp3')
        elif x=="cow":
            os.system('C:\\Users\\jeswa\\PycharmProjects\\Scarecrow\\sounds\\RunningSound.mp3')
        elif x=="bear":
            os.system('C:\\Users\\jeswa\\PycharmProjects\\Scarecrow\\sounds\\PanSound.mp3')
while True:
    outputnames = []
    success, img = cap.read()
    blob=cv2.dnn.blobFromImage(img,1/255,(wht,wht),[0,0,0],1,crop=False)
    net.setInput(blob)

    layer_names = net.getLayerNames()
    #print(layer_names)
    #print(net.getUnconnectedOutLayers())
    outputnames = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputnames)
    findobj(outputs,img)



    cv2.imshow('image',img)
    cv2.waitKey(1)
cap.release()
cv2.dertroyAllWindows()