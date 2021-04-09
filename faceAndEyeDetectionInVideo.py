# download cascade from CV2 Github
# https://github.com/opencv/opencv/tree/master/data/haarcascades

import cv2

face_cascade = cv2.CascadeClassifier('C:\Python Code\FacialDetection\FaceBasicDetection\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\Python Code\FacialDetection\FaceBasicDetection\haarcascade_eye_tree_eyeglasses.xml')

# capture a video file , any video file 

cap = cv2.VideoCapture('C:\Python Code\FacialDetection\FaceBasicDetection\smileFace.mp4')

while cap.isOpened():
    ret,img = cap.read()

    # convert the frame to gray
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #grab faces from each frame
    faces = face_cascade.detectMultiScale(gray,1.1,4)

    for (x,y,w,h) in faces:
        #draw a rectangle for each face
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)

        # for each face we will look for the eyes
        # we dont need to search the eyes for the whole frame
        # we can produce a region of interest

        roi_color = img[y:y+h , x:x+w ]
        roi_gray = gray[y:y+h , x:x+w ]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex,ey,ew,eh) in eyes:
            #draw rectangles for the eyes
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),4 )

    # show the frame
    cv2.imshow('img',img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()        




