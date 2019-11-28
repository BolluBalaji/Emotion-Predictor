import cv2
import os
import numpy as np
from keras.models import load_model
import dlib
from imutils import face_utils
import tensorflow as tf
path = os.getcwd()



IMG_WIDTH = IMG_HEIGHT = 48
emotions = ['Angry','Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']



print("Loading PreTrained Model......")

model_path = path + "\\" + 'Emotion_1.h5'
model = load_model(model_path)

file = path + "\\" + 'mmod_human_face_detector.dat'
print(file)

font = cv2.FONT_HERSHEY_SIMPLEX

#face_cascade = cv2.CascadeClassifier(file)
face_detect = dlib.get_frontal_face_detector()


print("Starting Web Cam....")
cap = cv2.VideoCapture(0)
print("Web Cam Started.....")
cap.set(3, 800) #WIDTH
cap.set(4, 500) #HEIGHT

while True:
    #Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = face_detect(gray,1)

    for (i,rect) in enumerate(rects):

        (x,y,w,h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        roi = frame[y:y+h, x:x+w]
        #cv2.imwrite("roi.jpg", roi)
        roi = cv2.resize(roi, (48, 48))
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        roi = np.array(roi) / 255.0

        roi = roi.reshape(-1,48,48,1)
        
        emotion = model.predict(roi)

        emotion = emotions[emotion.argmax()]
        
        text = "Emotion: " + str(emotion) 

        cv2.putText(frame,text,(x,y-10), font, 0.6, (255, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow('Age_Gender_Detector', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
