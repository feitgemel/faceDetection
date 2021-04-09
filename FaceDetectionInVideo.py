# download the casecase file from the CV2 git hub :
# https://github.com/opencv/opencv/tree/master/data/haarcascades

import cv2

face_Cascade = cv2.CascadeClassifier('c:\demo\haarcascade_frontalface_default.xml')

# you can use any video to capure the faces
cap = cv2.VideoCapture('C:\Python Code\FacialDetection\people.mp4')

while True:
    ret,img = cap.read()

    # convert to gray
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # grab faces from the gray image

    faces = face_Cascade.detectMultiScale(gray,1.1,4)

    # draw a rectangle for each face (for each frame)
    for (x,y,w,h) in faces :
        cv2.rectangle(img,(x,y), (x+w,y+h),(255,0,0),3)

    # show each image
    cv2.imshow('img',img)

    # break if 'q' pressed

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()


