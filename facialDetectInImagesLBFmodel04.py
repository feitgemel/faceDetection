import cv2

# get the image file
path = 'C:\Python Code\FacialDetection\GalGadot.jpg'
image = cv2.imread(path)

# resize the image
scale_percent = 60 # this the precent of the original image
width = int(image.shape[1]* scale_percent / 100 )
height = int (image.shape[0] * scale_percent / 100 )
dim = (width,height)

#resize the image
resizedImage = cv2.resize(image,dim , interpolation=cv2.INTER_AREA)

# convert to gray
imageGray = cv2.cvtColor(resizedImage,cv2.COLOR_BGR2GRAY)

# donwload the haar cascade from the cv2 Github 
# you can watch my previous videos for finding that file

harcascade = 'C:\Python Code\FacialDetection\haarcascade_frontalface_default.xml'
detector = cv2.CascadeClassifier(harcascade)
faces = detector.detectMultiScale(imageGray)

#load the LBF model file
# you can google it and download : "lbfmodel.yaml"

LBFModel = 'C:\Python Code\FacialDetection\lbfmodel.yaml'
landmarkDetector = cv2.face.createFacemarkLBF()
landmarkDetector.loadModel(LBFModel)

# get the position of the faces
_,landmarks = landmarkDetector.fit(imageGray,faces)

# count and draw the points
numerator=0
for landmark in landmarks:
    for x,y in landmark[0]:
        numerator=numerator+1
        cv2.circle(resizedImage,(x,y),1,(255,0,0),3)
        cv2.putText(resizedImage,str(numerator),(x,int(y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2 )


# display the image
cv2.imshow('resized image',resizedImage)
#cv2.imshow('gray image',imageGray)

cv2.waitKey(0)