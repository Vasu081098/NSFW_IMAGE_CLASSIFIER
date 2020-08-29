from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
# Keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

model = load_model('final_model.h5')
#model._make_predict_function()
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    img = img_to_array(img)
    # x = np.true_divide(x, 255)
    img = img.reshape(1,224,224,3)
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    preds = model.predict(img)
    return preds



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    classes = {'TRAIN':['NSFW','SFW'],
               'TEST' :['NSFW','SFW'],
               'VALIDATION' :['NSFW','SFW']}
    
    if request.method == 'POST':
	
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        arr = preds.argmax(axis=-1)
        # = str(arr[0])
        #return result
        predicted_class = classes['TRAIN'][arr[0]]
        print('This is {}.'.format(predicted_class))

        return str(predicted_class)
     

    return None
        


if __name__ == '__main__':
    app.run(debug=True)

