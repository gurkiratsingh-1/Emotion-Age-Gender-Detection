from pathlib import Path
import cv2
import sys
import numpy as np
from contextlib import contextmanager
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from os import listdir
from os.path import isfile, join
import dlib
#import os
classifier = load_model('./emotion_little_vgg_2.h5')
face_classifier = cv2.CascadeClassifier('./Haarcascades/haarcascade_frontalface_default.xml')
pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5"
            


def fun():
      


    modhash = 'fbe63257a054c1c5466cfd7bf14646d6'
    emotion_classes = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Surprise'}

    def face_detector(img):
        # Convert image to grayscale for faster detection
        gray = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        if faces is ():
            return False ,(0,0,0,0), np.zeros((1,48,48,3), np.uint8), img
        
        allfaces = []   
        rects = []
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi = img[y:y+h, x:x+w]
            allfaces.append(roi)
            rects.append((x,w,y,h))
        return True, rects, allfaces, img

    # Define our model parameters
    depth = 16
    k = 8
    weight_file = None
    margin = 0.4
    image_dir = None

    # Get our weight file 
    if not weight_file:
        weight_file = get_file("weights.28-3.73.hdf5", pretrained_model, cache_subdir="pretrained_models",
                               file_hash=modhash, cache_dir=Path(sys.argv[0]).resolve().parent)

    # load model and weights
    img_size = 64
    model = WideResNet(img_size, depth=depth, k=k)()
    model.load_weights(weight_file)

    # Initialize Webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        ret, rects, faces, image = face_detector(frame)
        preprocessed_faces_ag = []
        preprocessed_faces_emo = []
        
        if ret:
            for (i,face) in enumerate(faces):
                face_ag = cv2.resize(face, (64, 64), interpolation = cv2.INTER_AREA)
                preprocessed_faces_ag.append(face_ag)

                face_gray_emo = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_gray_emo = cv2.resize(face_gray_emo, (48, 48), interpolation = cv2.INTER_AREA)
                face_gray_emo = face_gray_emo.astype("float") / 255.0
                face_gray_emo = img_to_array(face_gray_emo)
                face_gray_emo = np.expand_dims(face_gray_emo, axis=0)
                preprocessed_faces_emo.append(face_gray_emo)
                
            # make a prediction for Age and Gender
            results = model.predict(np.array(preprocessed_faces_ag))
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()

            # make a prediction for Emotion 
            emo_labels = []
            for (i, face) in enumerate(faces):
                preds = classifier.predict(preprocessed_faces_emo[i])[0]
                emo_labels.append(emotion_classes[preds.argmax()])
            
            # draw results, for Age and Gender
            for (i, face) in enumerate(faces):
                label = "{}, {}, {}".format(int(predicted_ages[i]),
                                            "F" if predicted_genders[i][0] > 0.6 else "M",
                                            emo_labels[i])
                
            #Overlay our detected emotion on our pic
            for (i, face) in enumerate(faces):
                label_position = (rects[i][0] + int((rects[i][1]/2)), abs(rects[i][2] - 10))
                cv2.putText(image, label, label_position , cv2.FONT_HERSHEY_PLAIN,1, (0,255,0), 2)

        cv2.imshow("Emotion Detector", image)
        if cv2.waitKey(1) == 13: #13 is the Enter Key
            break

    cap.release()
    cv2.destroyAllWindows()


def fun1():
    image_path = "./images/"

    modhash = 'fbe63257a054c1c5466cfd7bf14646d6'
    emotion_classes = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Surprise'}
    
    def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=0.8, thickness=1):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
        
    
    # Define our model parameters
    depth = 16
    k = 8
    weight_file = None
    margin = 0.4
    image_dir = None
    
    # Get our weight file 
    if not weight_file:
        weight_file = get_file("weights.28-3.73.hdf5", pretrained_model, cache_subdir="pretrained_models",
                               file_hash=modhash, cache_dir=Path(sys.argv[0]).resolve().parent)
    # load model and weights
    img_size = 64
    model = WideResNet(img_size, depth=depth, k=k)()
    model.load_weights(weight_file)
    
    detector = dlib.get_frontal_face_detector()
    
    image_names = [f for f in listdir(image_path) if isfile(join(image_path, f))]
    
    for image_name in image_names:
        frame = cv2.imread("./images/" + image_name)
        preprocessed_faces_emo = []           
     
        input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)
        detected = detector(frame, 1)
        faces = np.empty((len(detected), img_size, img_size, 3))
        
        preprocessed_faces_emo = []
        if len(detected) > 0:
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - margin * w), 0)
                yw1 = max(int(y1 - margin * h), 0)
                xw2 = min(int(x2 + margin * w), img_w - 1)
                yw2 = min(int(y2 + margin * h), img_h - 1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                faces[i, :, :, :] = cv2.resize(frame[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
                face =  frame[yw1:yw2 + 1, xw1:xw2 + 1, :]
                face_gray_emo = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_gray_emo = cv2.resize(face_gray_emo, (48, 48), interpolation = cv2.INTER_AREA)
                face_gray_emo = face_gray_emo.astype("float") / 255.0
                face_gray_emo = img_to_array(face_gray_emo)
                face_gray_emo = np.expand_dims(face_gray_emo, axis=0)
                preprocessed_faces_emo.append(face_gray_emo)
    
            # make a prediction for Age and Gender
            results = model.predict(np.array(faces))
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()
    
            # make a prediction for Emotion 
            emo_labels = []
            for i, d in enumerate(detected):
                preds = classifier.predict(preprocessed_faces_emo[i])[0]
                emo_labels.append(emotion_classes[preds.argmax()])
            
            # draw results
            for i, d in enumerate(detected):
                label = "{}, {}, {}".format(int(predicted_ages[i]-8),
                                            "F" if predicted_genders[i][0] > 0.4 else "M", emo_labels[i])
                draw_label(frame, (d.left(), d.top()), label)
    
        cv2.imshow("Emotion Detector", frame)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()




