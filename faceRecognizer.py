import cv2
import os
import numpy as np
from PIL import Image

haarCascadeDest = "haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(haarCascadeDest)

faceRecognizer = cv2.face.createLBPHFaceRecognizer()
def names(nbractual, confidence):
    if nbractual == 1:
        print("\tSubject Selim is correctly recognized with confidence of {} precent.".format(round(confidence,2)))
    elif nbractual == 2:
        print("\tSubject Rabia is correctly recognized with confidence of {} precent.".format(round(confidence,2)))
    elif nbractual == 3:
        print("\tSubject Sinan is correctly recognized with confidence of {} precent.".format(round(confidence,2)))
    elif nbractual == 4:
        print("\tSubject Mehmet is correctly recognized with confidence of {} precent.".format(round(confidence,2)))
    elif nbractual == 5:
        print("\tSubject Onur is correctly recognized with confidence of {} precent.".format(round(confidence,2)))
    elif nbractual == 6:
        print("\tSubject Engin is correctly recognized with confidence of {} precent.".format(round(confidence,2)))


def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path) if not f.endswith('.png')] 
    images = [] 
    labels = []

    print("\n\n\tAdding faces to training set...")
    for imagePath in imagePaths:
        imagePil = Image.open(imagePath).convert('L')
        image = np.array(imagePil,'uint8')
        nbr = int(os.path.split(imagePath)[1].split(".")[0].replace("subject",""))
        faces = faceCascade.detectMultiScale(image,1.03,1,0)
        for (x,y,w,h) in faces:
            images.append(image[y:y+h,x:x+w])
            labels.append(nbr)
            cv2.imshow("Adding faces to training set...",image[y:y+h,x:x+w])

            cv2.waitKey(50)


    return images, labels

average = 0.0
recognized = 0
total = 0
path = 'profiles'
images, labels = getImagesAndLabels(path)
cv2.destroyAllWindows()

faceRecognizer.train(images,np.array(labels))
print("\t\tNow our recognizer will try to recognize faces...\n")
imagePaths = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.png')]
for imagePath in imagePaths:
    predictImagePil = Image.open(imagePath).convert('L')
    predictImage = np.array(predictImagePil,'uint8')
    faces = faceCascade.detectMultiScale(predictImage,1.03,1,0)
    for (x,y,w,h) in faces:
        nbrPredicted, conf = faceRecognizer.predict(predictImage[y:y+h,x:x+w])
        nbrActual = int(os.path.split(imagePath)[1].split(".")[0].replace("subject",""))
        total+=1
        if nbrActual == nbrPredicted:
           names(nbrActual,conf)
           recognized+=conf
        else:
            print ("\t{} is incorrectly recognized as{}".format(nbrActual,nbrPredicted))
        cv2.imshow("Recognizing Face", predictImage[y:y+h,x:x+w])
        cv2.waitKey(1000)


average = recognized/total


print("\n\t\tAll faces are recognized with the percentage displayed above.")
print("\tAverage prediction ",round(average,2))

