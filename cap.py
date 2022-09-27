import numpy as np
import cv2

#face_cascade = cv2.CascadeClassifier('/usr/local/opencv/data/haarcascades/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('/usr/local/opencv/data/haarcascades/haarcascade_eye.xml')

img1 = cv2.imread('1.jpg')          # queryImage
#img2 = cv2.imread('2.jpg') # trainImage

gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

#faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#for (x,y,w,h) in faces:
#    cv2.rectangle(img1,(x,y),(x+w,y+h),(255,0,0),2)
#    roi_gray = gray[y:y+h, x:x+w]
#    roi_color1 = img1[y-100:y+h, x-40:x+w]
#   roi_color2 = img2[y-100:y+h, x-40:x+w]
eyes1 = eye_cascade.detectMultiScale(img1)
#    eyes2 = eye_cascade.detectMultiScale(roi_color2)
for (ex,ey,ew,eh) in eyes1:
    cv2.rectangle(img1,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    roi_color3 = img1[0:ey, 0:ex+ex]
#   for (ex,ey,ew,eh) in eyes2:
# 	cv2.rectangle(roi_color2,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
#	roi_color4 = roi_color2[0:ey, 0:x+w]

	
cv2.imshow('img1',roi_color3)
#cv2.imshow('img2',roi_color4) 
cv2.waitKey(0)
cv2.destroyAllWindows()
