#python drowniness_yawn.py --webcam webcam_index
#import RPi.GPIO as GPIO
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os
import urllib.request as urllib
import winsound
import subprocess


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance
def jaw(shape):
    jaw=face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
def nose(shape):
    nose=face_utils.FACIAL_LANDMARKS_IDXS["nose"]
def right_eyebrow(shape):
    right_eyebrow=face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
def left_eyebrow(shape):
    left_eyebrow=face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]


ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_imgS = 3

alarm_status = False
alarm_status2 = False
saying = False
YAWN_THRESH = 10
COUNTER = 0
num=0
TOTAL = 0
print("-> Loading the predictor and detector...")
#detector = dlib.get_frontal_face_detector()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")    #Faster but less accurate
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


print("-> Starting Video Stream")
#url="http://192.168.43.1:8080/shot.jpg"
vs = VideoStream(src=args["webcam"]).start()
#vs= VideoStream(usePiCamera=True).start()       //For Raspberry Pi
time.sleep(1.0)

while True:
##    imgPath=urllib.urlopen(url)
##    imgNp=np.array(bytearray(imgPath.read()),dtype=np.uint8)
##    img=cv2.imdecode(imgNp,-1)
    img = vs.read()
    img = imutils.resize(img, width=450)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #rects = detector(gray, 0)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)

    #for rect in rects:
    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
        
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        

        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye [1]
        rightEye = eye[2]

        distance = lip_distance(shape)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)
        jaw = shape[0:17]
        cv2.drawContours(img, [jaw], -1, (0, 255, 0), 1)
        
        
        lip = shape[48:60]
        cv2.drawContours(img, [lip], -1, (0, 255, 0), 1)
        right_eyebrow = shape[17:22]
        cv2.drawContours(img, [right_eyebrow], -1, (0, 255, 0), 1)
        left_eyebrow = shape[22:27]
        cv2.drawContours(img, [left_eyebrow], -1, (0, 255, 0), 1)
        nose = shape[27:35]
        cv2.drawContours(img, [nose], -1, (0, 255, 0), 1)
        

        
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_imgS:
                
                winsound.Beep(1500,1000)
                f=open('log.txt','w')
                f.write("***DROWSINESS ALERT!***")
                f.close()  
                subprocess.Popen('python sms.py',shell=True).communicate()
                cv2.putText(img, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if (distance > YAWN_THRESH):
                cv2.putText(img, "Yawn Alert", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                winsound.Beep(1000,500)
                f=open('log.txt','w')
                f.write("***yawn alert***")
                f.close()  
                subprocess.Popen('python sms.py',shell=True).communicate()
                cv2.putText(img, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                

          

        cv2.putText(img, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, "YAWN: {:.2f}".format(distance), (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                   

    cv2.imshow("frame", img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
