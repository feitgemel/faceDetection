# links for download dlib library
#https://pypi.org/simple/dlib/
# works with python 3.6

#install dlib python package 
#https://www.youtube.com/watch?v=HqjcqpCNiZg

import cv2
import numpy as np 
import dlib 
from math import hypot

cap = cv2.VideoCapture(0)
noseImage = cv2.imread('C:\Python-Code\FacialDetection\RemoveEyes\clown-nose-no-background-free-removebg.png')
eye1Image = cv2.imread('C:\Python-Code\FacialDetection\RemoveEyes\BiaELAeaT1last.png')
eye2Image = cv2.imread('C:\Python-Code\FacialDetection\RemoveEyes\BiaELAeaT2last.png')


cap.set(3,1920)
cap.set(4,1080)


detector = dlib.get_frontal_face_detector()

# we need the file "shape_predictor_68_face_landmarks.dat"
#==========================================================
predictor = dlib.shape_predictor('C:\Python-Code\FacialDetection\shape_predictor_68_face_landmarks.dat')
eyeYpositionParam=110
widthMulpileParam=3.8

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    #print(faces)
    for face in faces:
        landmarks = predictor(gray,face)
        
        # EYE1
        # ====
        Eye1Left = (landmarks.part(42).x , landmarks.part(42).y)
        Eye1Right =(landmarks.part(45).x , landmarks.part(45).y)
        Eye1Width = int(hypot(Eye1Left[0] - Eye1Right[0], Eye1Left[1] - Eye1Right[1]))  
        Eye1Width = int(Eye1Width * widthMulpileParam)
        Eye1Height = int(Eye1Width)# * 1.07) 
        
        eye1Image = cv2.resize(eye1Image,(Eye1Width,Eye1Height))
        # draw rectangle for the Eye1
        eye1TopLeft = (int(landmarks.part(42).x-30),int(landmarks.part(44).y-eyeYpositionParam))
        eye1BottomRight = (int(landmarks.part(45).x),int(landmarks.part(46).y-eyeYpositionParam))
        # only show for the step
        #cv2.rectangle(frame,eye1TopLeft,eye1BottomRight,(0,255,0),2 )
        eye1Area = frame[eye1TopLeft[1]:eye1TopLeft[1]+Eye1Height , eye1TopLeft[0]:eye1TopLeft[0]+Eye1Width]
 
        eye1ImageGray = cv2.cvtColor(eye1Image,cv2.COLOR_BGR2GRAY)
        _, eye1Mask = cv2.threshold(eye1ImageGray,25,255,cv2.THRESH_BINARY_INV) 
        eye1AreawithoutTheEye1 = cv2.bitwise_and(eye1Area,eye1Area, mask=eye1Mask)
        
        finalEye1 = cv2.add(eye1AreawithoutTheEye1,eye1Image ) 
        #display the finalEye 

        #step9
        # put it inside
        frame[eye1TopLeft[1]:eye1TopLeft[1]+Eye1Height , eye1TopLeft[0]:eye1TopLeft[0]+Eye1Width] = finalEye1

        #--------------------------------------------------------
        # EYE2
        # ====
        Eye2Left = (landmarks.part(36).x , landmarks.part(36).y)
        Eye2Right =(landmarks.part(39).x , landmarks.part(39).y)
        Eye2Width = int(hypot(Eye2Left[0] - Eye2Right[0], Eye2Left[1] - Eye2Right[1]))  
        Eye2Width = int(Eye2Width * widthMulpileParam)
        Eye2Height = int(Eye2Width) #* 1.07) 

        eye2Image = cv2.resize(eye2Image,(Eye2Width,Eye2Height))

        # draw rectangle for the Eye2
        eye2TopLeft = (int(landmarks.part(36).x-80),int(landmarks.part(38).y-eyeYpositionParam))
        eye1BottomRight = (int(landmarks.part(39).x),int(landmarks.part(40).y-eyeYpositionParam))
        
        #cv2.rectangle(frame,eye2TopLeft,eye2BottomRight,(0,255,0),2 )
        eye2Area = frame[eye2TopLeft[1]:eye2TopLeft[1]+Eye2Height , eye2TopLeft[0]:eye2TopLeft[0]+Eye2Width]
 
        eye2ImageGray = cv2.cvtColor(eye2Image,cv2.COLOR_BGR2GRAY)
        _, eye2Mask = cv2.threshold(eye2ImageGray,25,255,cv2.THRESH_BINARY_INV) 
        eye2AreawithoutTheEye2 = cv2.bitwise_and(eye2Area,eye2Area, mask=eye2Mask)
        
        finalEye2 = cv2.add(eye2AreawithoutTheEye2,eye2Image ) 
        #display the finalEye2 

        # put eye2 it inside
        frame[eye2TopLeft[1]:eye2TopLeft[1]+Eye2Height , eye2TopLeft[0]:eye2TopLeft[0]+Eye2Width] = finalEye2

        #--------------------------------------------------------


        
        topNose =  (landmarks.part(29).x , landmarks.part(29).y)
        leftNose = (landmarks.part(31).x , landmarks.part(31).y)
        rightNose =(landmarks.part(35).x , landmarks.part(35).y)
        # this is step1
        # in the first step show the eye or nose  
        #cv2.circle(frame,topNose,3,(255,0,0),-1)

        # this is step 2
        noseWidth = int(hypot(leftNose[0] - rightNose[0], leftNose[1] - rightNose[1])*1.5)  
        #noseWidth = int(landmarks.part(35).x) - int(landmarks.part(31).x ) 
        noseWidth = int(noseWidth*1.5)
        noseHeight = int(noseWidth * 0.81) 
        #propotion of the image : 344/321 = 1.07
        #noseHeight = int(noseWidth * 1.336)
        
        #print (noseWidth)
        #print (noseHeight)
        
        #step 3.1
        # show the image of the eye
         
        #step 3.2 
        # resize the eye 
        noseImage = cv2.resize(noseImage,(noseWidth,noseHeight))
        
        #print(noseWidth)
        
        # step4 place it on the area
        centerNose = leftNose = (landmarks.part(30).x , landmarks.part(30).y)
        # draw rectangle for the nose
        noseTopLeft = (int(landmarks.part(31).x-30),int(landmarks.part(30).y-30))
        noseBottomRight = (int(landmarks.part(35).x),int(landmarks.part(33).y))
        # only show for the step
        #cv2.rectangle(frame,noseTopLeft,noseBottomRight,(0,255,0),2 )

        #step 5
        # get the nose area
        # we dony anymore to show the eye pic
        noseArea = frame[noseTopLeft[1]:noseTopLeft[1]+noseHeight , noseTopLeft[0]:noseTopLeft[0]+noseWidth]
     

        #step 6 
        noseImageGray = cv2.cvtColor(noseImage,cv2.COLOR_BGR2GRAY)
        _, noseMask = cv2.threshold(noseImageGray,25,255,cv2.THRESH_BINARY_INV) 
        # show the mask - three images togheter

        #step 7
        noseAreawithoutTheNose = cv2.bitwise_and(noseArea,noseArea, mask=noseMask)
        #display the eyeAreawithoutTheEye
        
        #step 8
        # Both images togheter 
        finalNose = cv2.add(noseAreawithoutTheNose,noseImage ) 
        #display the finalEye 

        #step9
        # put it inside
        frame[noseTopLeft[1]:noseTopLeft[1]+noseHeight , noseTopLeft[0]:noseTopLeft[0]+noseWidth] = finalNose


    cv2.imshow('frame',frame) 
    #cv2.imshow('nose',noseImage)
    #cv2.moveWindow('nose',100,0)
    #cv2.imshow('noseMask',noseMask)
    #cv2.moveWindow('noseMask',250,0)
    #cv2.imshow('nose Area', noseArea)
    #cv2.moveWindow('nose Area',400,0) 

    #cv2.imshow('noseAreawithoutTheEye', noseAreawithoutTheNose)
    #cv2.moveWindow('noseAreawithoutTheEye',550,0) 

    #cv2.imshow('finalNose', finalNose)
    #cv2.moveWindow('finalNose',700,0) 

    #cv2.imshow('eye2Image', eye2Image)


    

    key = cv2.waitKey(1)
    if key == ord('q'):
        break


