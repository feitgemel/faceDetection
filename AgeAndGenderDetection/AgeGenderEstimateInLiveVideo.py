# download the pre-trained models
# https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/

import cv2
import matplotlib.pyplot as plt 

age_model = cv2.dnn.readNetFromCaffe("C:/Python-Code/FacialDetection/AgeAndGenderDetection/age.prototxt","C:/Python-Code/FacialDetection/AgeAndGenderDetection/dex_chalearn_iccv2015.caffemodel")
gender_model = cv2.dnn.readNetFromCaffe("C:/Python-Code/FacialDetection/AgeAndGenderDetection/gender.prototxt","C:/Python-Code/FacialDetection/AgeAndGenderDetection/gender.caffemodel")


detector = cv2.CascadeClassifier('C:\Python-Code\FacialDetection\AgeAndGenderDetection\haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)
cap.set(3,1920)
cap.set(4,1080)

while cap.isOpened():

    re, img = cap.read()
    estimatedAge = 0

    faces = detector.detectMultiScale(img, 1.3 , 5)
    x , y , w , h = faces[0]

    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    detect_face_roi = img[y:y+h , x:x+w ]

    # the mode need a image of 244X244
    # resize it 
    detect_face_roi = cv2.resize(detect_face_roi,(224,224))

    print ('original shape:',detect_face_roi.shape)

    # but the mode need the shape in the reverse direction
    detect_face_blob = cv2.dnn.blobFromImage(detect_face_roi)
    print ('after blob shape:',detect_face_blob.shape)


    gender_model.setInput(detect_face_blob)
    gender_result = gender_model.forward()


    print(gender_result[0])


    womanEstimate = round(gender_result[0][0])
    manEstimate = round(gender_result[0][1])
    print('womanEstimate',womanEstimate)
    print('manEstimate',manEstimate)


    if womanEstimate == 1 :
        print('Woman')
        cv2.rectangle(img, (20, 50), (300, 100), (255, 255, 0), cv2.FILLED)
        cv2.putText(img, "gender : Woman", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
    else:
        print('Man')
        cv2.rectangle(img, (20, 50), (300, 100), (255, 255, 0), cv2.FILLED)
        cv2.putText(img, "gender : Man", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)


    age_model.setInput(detect_face_blob)
    age_result = age_model.forward()

    length_of_result = len(age_result[0])
    #print('length_of_result:',length_of_result )
    #print('age results:' , age_result[0])

    # multiplay each index and then do a sum
    for i in range (0,length_of_result):
        #print('i',i,'age:',age_result[0][i], 'X : ', i*age_result[0][i] )
        #estimatedAge = estimatedAge + (i*age_result[0][i])
        #estimatedAge = round(estimatedAge)
        #ageText = 'age : ' + str(estimatedAge) + ' years old'

    print (estimatedAge,'estimatedAge')
    cv2.rectangle(img, (20, 150), (350, 200), (255, 255, 0), cv2.FILLED)
    cv2.putText(img, ageText , (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)


        
        
    #apparent_age = round(np.sum(age_result[0]*indexes))

    #print('apparent_age:' , apparent_age , 'years old')


    cv2.imshow('img',img)
    cv2.imshow('detect_face_roi',detect_face_roi)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 


