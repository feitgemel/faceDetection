# detect face emotions in images

import cv2

# You should install deepface library
# pip install deepface
# this code was tested in Python 3.8

from deepface import DeepFace


# you can read any face image
# lets try a diffrent image :

#img = cv2.imread('C:\Python Code\FaceEmotion\youngwoman.jpg')
img = cv2.imread('faceDetection/woman.jpg')

#find the face
# you can find the file in the OpenCV Github page 
face_cascade = cv2.CascadeClassifier('faceDetection/haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray,1.1,4)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)

obj = DeepFace.analyze(img_path = img , actions = ['emotion'], enforce_detection=False)
print(obj)

emotion =  obj[0]['dominant_emotion']

print(emotion)

txt = 'Emotion: ' + str(emotion)

#lets add it to the image 
cv2.putText(img, txt, (50,100),cv2.FONT_HERSHEY_COMPLEX, 1 , (0,0,255), 3 )

#show the image
cv2.imshow('img',img)
cv2.waitKey()

cv2.destroyAllWindows()