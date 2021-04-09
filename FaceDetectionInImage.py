import cv2

#Download the haarcascades
#https://github.com/opencv/opencv/tree/master/data/haarcascades
# copy the file to the Python code direcotory

face_cascade = cv2.CascadeClassifier('c:\demo\haarcascade_frontalface_default.xml')

# you cad use any facial image
img = cv2.imread('C:\Python Code\FacialDetection\GalGadot.jpg')
#Resize 
scale_percent = 60 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

resizedImage = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

# change the image to gray image
gray = cv2.cvtColor(resizedImage,cv2.COLOR_BGR2GRAY)

#look for faces in the gray image
faces = face_cascade.detectMultiScale(gray,1.1,4)

# get the image coordinates

for (x,y,w,h) in faces:
    #draw a rectangle 
    cv2.rectangle(resizedImage, (x,y),(x+w,y+h),(255,0,0),3)

# show the image
cv2.imshow('img',resizedImage)
cv2.waitKey()