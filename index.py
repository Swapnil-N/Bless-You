import cv2
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import numpy as np

import matplotlib.pyplot as plt
import PIL.Image as Image

from category import Category

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

frames = []

counter = 0

# https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub
model_path = './saved_models/1599964712'
model = tf.keras.models.load_model(model_path)

IMAGE_SHAPE = (224, 224)

categories = [Category.NORMAL, Category.SCRATCHING, Category.SNEEZING]


def processFrame(frame):
    # model.predict
    frames.append(frame)
    resized = Image.fromarray(frame).resize(IMAGE_SHAPE)
    npimg = np.array(resized)/255.0
    predictions = model.predict(npimg[np.newaxis, ...]))


while True:
    # Grab a single frame of video
    ret, frame=video_capture.read()

    # Resize frame of video to 1/4 size for easier compute if needed
    # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Only process frames depending on counter of video to save space and compute
    if counter == 30:
        processFrame(frame)
        counter=0
    counter += 1

    # Display each frame AKA the video
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


for item in frames:  # shows all the captures frames
    cv2.imshow('pic', item)
    cv2.waitKey(0)  # press any key to advance


# Release handle to the webcam
# video_capture.release()
# cv2.destroyAllWindows()
