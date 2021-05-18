# you can follow and download this pretrained model :
# https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/

import cv2

ageModel = cv2.dnn.readNetFromCaffe('C:/GitHub/faceDetection/AgeAndGenderDetection/age.prototxt','C:/GitHub/faceDetection/AgeAndGenderDetection/dex_chalearn_iccv2015.caffemodel')
genderModel = cv2.dnn.readNetFromCaffe('C:/GitHub/faceDetection/AgeAndGenderDetection/gender.prototxt','C:/GitHub/faceDetection/AgeAndGenderDetection/gender.caffemodel')

estimateAge = 0

#img = cv2.imread('C:/GitHub/faceDetection/AgeAndGenderDetection/gal2.jpg')
img = cv2.imread('C:/GitHub/faceDetection/AgeAndGenderDetection/steve_jobs.jpg')

detector = cv2.CascadeClassifier('C:/GitHub/faceDetection/AgeAndGenderDetection/haarcascade_frontalface_default.xml')
faces = detector.detectMultiScale(img, 1.3, 5)

#find the face inside the image
x , y, w , h = faces[0]

x = int(x)
y = int(y)
w = int(w)
h = int(h)

#  get the region of interest

detectFaceRoi = img[y:y+h , x:x+w]

#resize the image to the model requirment 224 X 224
detectFaceRoi = cv2.resize(detectFaceRoi,(224,224))

#lets print the shape to test it 
print ('before convert detectFaceRoi shape:',detectFaceRoi.shape) 

# the model need the oposite shape 
detectFaceBlob = cv2.dnn.blobFromImage(detectFaceRoi)
print ('detectFaceBlob shape:',detectFaceBlob.shape) 

genderModel.setInput(detectFaceBlob)
genderResult = genderModel.forward()

print(genderResult[0])

# watch the result . if the first item is 1 then it is a woman , else it is a man

womanEstimate = round(genderResult[0][0])
manEstimate = round(genderResult[0][1])  

print ('womanEstimate',womanEstimate)
print ('manEstimate',manEstimate)

if womanEstimate == 1:
    cv2.rectangle(img, (20,50),(300,100),(255,255,0),cv2.FILLED)
    cv2.putText(img, 'gender : Woman', (30,80), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 4)
else:
    cv2.rectangle(img, (20,50),(300,100),(255,255,0),cv2.FILLED)
    cv2.putText(img, 'gender : Man', (30,80), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 4)


# age estimate 
ageModel.setInput(detectFaceBlob)
ageResult = ageModel.forward()

lengthOfResult = len(ageResult[0])

for i in range (0,lengthOfResult):
    estimateAge = estimateAge + (i * ageResult[0][i])
    estimateAge = round(estimateAge)
    ageText = 'age : ' + str(estimateAge) + ' years old'

print('estimateAge: ', estimateAge)

cv2.rectangle(img, (20,150),(350,200),(255,255,0),cv2.FILLED)
cv2.putText(img, ageText , (30,180), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 4)


cv2.imshow('img',img)
cv2.imshow('detectFaceRoi',detectFaceRoi)

#add a wait key
cv2.waitKey()







