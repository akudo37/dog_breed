#!/usr/bin/env python3
import numpy as np
import cv2
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.applications.resnet50 import ResNet50, preprocess_input
import pickle
from PIL import Image

# global variables
face_cascade = None
ResNet50_model = None
model = None
ResNet50_base = None
dogNames = None


# Utitity
# path to tensor functions
def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img_path.seek(0)
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


# Human face detector
# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    global face_cascade
    if face_cascade is None:
        face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_alt.xml')

    img_path.seek(0)
    pil_img = Image.open(img_path)
    file_bytes = np.asarray(pil_img, dtype=np.uint8)
    img = cv2.cvtColor(file_bytes, cv2.COLOR_RGBA2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    return len(faces) > 0


# Add rectangles on detected faces in image
def add_face_rectangles(img_path):
    global face_cascade
    if face_cascade is None:
        face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_alt.xml')

    img_path.seek(0)
    pil_img = Image.open(img_path)
    file_bytes = np.asarray(pil_img, dtype=np.uint8)
    img = cv2.cvtColor(file_bytes, cv2.COLOR_RGBA2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    # get bounding box for each detected faces
    for (x,y,w,h) in faces:
        # add bouding box to color image
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 5)

    return img


# Dog detector
def ResNet50_predict_labels(img_path):
    global ResNet50_model
    if ResNet50_model is None:
        ResNet50_model = ResNet50(weights='imagenet')

    # returns prediction vector for image located at img_path
    print('[INFO:] Detecting...')
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))


# Dog breed classifier
def predict_dog_breed(image_path):
    global model
    global ResNet50_base
    global dogNames

    if model is None:
        ### TODO: Define your architecture.
        model = Sequential()
        model.add(Flatten(input_shape=(1,1,2048)))
        model.add(Dense(units=256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units=133, activation='softmax'))

        ### TODO: Load the model weights with the best validation loss.
        model.load_weights('./models/weights.best.Resnet50.hdf5')

    if ResNet50_base is None:
        ### TODO: Write a function that takes a path to an image as input
        ### and returns the dog breed that is predicted by the model.
        ResNet50_base = ResNet50(weights='imagenet', include_top=False)

    if dogNames is None:
        # load dog name list
        with open('./data/dogNames.pickle', 'rb') as f:
            dogNames = pickle.load(f)

    input_tensor = ResNet50_base.predict(preprocess_input(path_to_tensor(image_path)))
    return dogNames[np.argmax(model.predict(input_tensor))]


# Dog or dog resemblance classifier
def classify_dog(img_path):
    '''
    INPUT:
    image_path - (string) filepath of uploaded image file

    OUTPUT:
    [status string] - (string) 'dog', 'face', or 'neither'
    [dog breed] - (string) Dog breed name, or None
    '''
    if dog_detector(img_path):
        print('[INFO:] Dog is detected.')
        return 'dog', predict_dog_breed(img_path)

    if face_detector(img_path):
        print('[INFO:] Face is detected.')
        return 'face', predict_dog_breed(img_path)

    print('[INFO:] Neither dog nor face is detected.')
    return 'neither', None
