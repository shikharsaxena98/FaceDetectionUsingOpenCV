import numpy as np
import cv2

print("Please Enter your Face ID\n")
face_id=input()
print("[INFO] Please Wait and look in the Camera")

cap=cv2.VideoCapture(0)
face_detector=cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')


count=0
while(True):
    ret,img=cap.read()
    #cv2.imshow('img',img)
    #img=cv2.flip(img,1,-1)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('imgflip',img)
    faces=face_detector.detectMultiScale(gray,3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1
        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)
    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k==27 or k==97:
        break
    elif count >= 30: # Take 30 face sample and stop video
         break


    
cap.release()
cv2.destroyAllWindows()
