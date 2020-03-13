import cv2
import numpy as np
from time import sleep
from keras.models import load_model

train_model =  "ResNet" 

# Size of the images
if train_model == "ResNet":
    img_width, img_height = 197, 197

emotions = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Reinstantiate the fine-tuned model (Also compiling the model using the saved training configuration (unless the model was never compiled))
model = load_model('./../trained_models/ResNet-50.h5')

# Create a face cascade
cascPath = './../trained_models/haarcascade_frontalface_alt.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

def preprocess_input(image):
    image = cv2.resize(image, (img_width, img_height))  
    ret = np.empty((img_height, img_width, 3)) 
    ret[:, :, 0] = image
    
    ret[:, :, 1] = image
    ret[:, :, 2] = image
    x = np.expand_dims(ret, axis = 0)   

    if train_model == "ResNet":
        x -= 128.8006
        x /= 64.6497    

    return x

def predict(emotion):
    prediction = model.predict(emotion)
    
    return prediction

while True:
    if not video_capture.isOpened():	
        print('Unable to load camera.')
        sleep(5)						
    else:
        sleep(0.5)
        ret, frame = video_capture.read()						
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)	
        
        faces = faceCascade.detectMultiScale(gray_frame,scaleFactor= 1.1,minNeighbors= 5, minSize= (30, 30))

        prediction = None
        x, y = None, None

        for (x, y, w, h) in faces:
            ROI_gray = gray_frame[y:y+h, x:x+w] # Extraction of the region of interest (face) from the frame

            #face_rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)

            emotion = preprocess_input(ROI_gray)
            prediction = predict(emotion)
            #print(prediction[0][0])
            top_1_prediction = emotions[np.argmax(prediction)]

            #output_text
            cv2.putText(frame, top_1_prediction, (x, y+(h+50)), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA)

        # Display the resulting frame
        frame = cv2.resize(frame, (800, 500))
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q') :
            break

# quit
video_capture.release()
cv2.destroyAllWindows()
