import cv2
import sys


# link to path (im gae & haar cascade
img_path="C:\\Users\\Arslan\\Desktop\\Single_Person.jpg"
casc_path="C:\\Users\\Arslan\\Desktop\\Python\\face_detection\\haarcascade_frontalface_default.xml"

#face haar cascade creation 
faceCascade=cv2.CascadeClassifier(casc_path)


# read the img ad convert it into gray scale img
img= cv2.imread(img_path)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


# now we need to do the main task that is to detect faces in imgs

faces=faceCascade.detectMultiScale(
gray,
scaleFactor=1.1,
minNeighbors=5,
minSize=(30,30),
#flags=cv2.cv.CV_HAAR_SCALE_IMGAE # excluded in python 3.x

)


#now we have detected faces and we need to display it 

print("Total no of faces are :" .format(len(faces)))

#draw a rectangle around faces

for (x,y,w,h) in faces:

 cv2.rectangle(img, (x,y), (x+w, y+h), (0,255, 0),2)

# show the image with rectangle around the face
cv2.imshow("Faces found", img)

cv2.waitkey(0)
