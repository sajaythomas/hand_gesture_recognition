#coding=utf-8

from crypt import methods
import numpy as np
import os
import sys
import glob
import re

# Keras
import tensorflow as tf
from tensorflow. keras. applications. imagenet_utils import preprocess_input, decode_predictions
from tensorflow. keras. models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# Flask utils
from flask import Flask, redirect, url_for, request, render_template

# Define a flask app
app = Flask (__name__)

model_path = 'transfer_learning_efficientnetb5_sd.h5'

# Load Model
model = load_model(model_path)
# model.make_predict_function()

# # Preprocessing Function
# def model_predict (img_path, model):
#     img = image.load_img(img_path, target_size=(224, 224))
#     # Preprocessing the image
#     x = image.img_to_array(img)
#     # X = np. true divide(x, 255)
#     x = np.expand_dims(x, axis=0)
#     # Be careful how your trained model deals with the input
#     # otherwise, it won't make correct prediction!
#     x = preprocess_input(x)
    
#     preds = model.predict(x)
#     return preds

def model_predict (img_path, model):
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    return score


    


@app.route('/', methods = ['GET'])
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods = ['GET', 'POST'])
def upload():
    if request.method == "POST":
        # Get file from Post
        f=request.files['file']
        # Save the file to /uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename)
        )
        f.save(file_path)
        # Make Prediction
        predict = model_predict(file_path, model)
        # pred_class = decode_predictions(pred, top=1)
        CLASSES=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 
                'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        # pred_class = CLASSES[np.argmax(pred)]
        gestures = CLASSES[np.argmax(predict)]
        # result = str(pred_class[0][0][1])
        # return result
        return gestures
        

    return None

if __name__ =='__main__':
    app.run(debug=True)
