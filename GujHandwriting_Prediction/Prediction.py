from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/Dataset_Sample.h5'

# Load your trained model
model = load_model(MODEL_PATH)

print('Model loaded. Check http://127.0.0.1:5000/')

class_mappings={'૬':0,'૧':1,'ક':2,'ઔ':3,'શે':4,'ખ':5}


def model_predict(img_path, model):
    test_image = image.load_img(img_path, target_size=(200, 200))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image,axis=0)
    test_image = preprocess_input(test_image)
    result = model.predict(test_image)
    return result

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('UI.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
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
        predicted_class_num = np.argmax(preds)
        predicted_class = list(class_mappings.keys())[list(class_mappings.values()).index(predicted_class_num)]
        return str(predicted_class)
    return None


if __name__ == '__main__':
    app.run(debug=True)