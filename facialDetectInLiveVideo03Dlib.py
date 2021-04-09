# How to detect face landmarks whithin a live video camera using Dlib library
#============================================================================

# install the dlib library from this link :
#https://pypi.org/simple/dlib/
# Please notice : It works with Python 3.6 !!!!

import cv2
import numpy as np 
import dlib 

# setting the camera + image resolution
cap = cv2.VideoCapture(0) 
cap.set(3,1280)
cap.set(4,720)

detector = dlib.get_frontal_face_detector()

#We need the 68 predictor weights
#This is the file : "shape_predictor_68_face_landmarks.dat"
# You can google it and download

predictor = dlib.shape_predictor('C:\Python Code\FacialDetection\shape_predictor_68_face_landmarks.dat')


while True:
    _, frame = cap.read()

    # show the image
    cv2.imshow('frame',frame)

    #convert the frame to gray
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # grab the faces
    faces = detector(gray)

    # print the "faces"
    #print(faces)

    # grab each face using a loop
    for face in faces:
        # we have the coordinate of each face 
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        # draw a rectangle over the face
        #cv2.rectangle(frame,(x1,y1), (x2,y2) , (0,255,0) , 3)

        # Grab the landmakrs for each face (the 68 points)
        landmakrs = predictor(gray,face)
        #print(landmakrs)

        for n in range (0 , 68): # since the model has 68 landmark points
            x = landmakrs.part(n).x
            y = landmakrs.part(n).y 
            # draw a point 
            cv2.circle(frame,(x,y),3,(255,0,0),-1)
            
    cv2.imshow('frame',frame) 

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


