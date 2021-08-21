# coding=utf-8
from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template,Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
model = load_model('models/my_model')
#MODEL_PATH = 'models/chestv-Copy.h5'
# Load your trained model
# model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
print('Model loaded. Check https://')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    #x = x/255
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='tf')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


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
        print(preds)
        # Process your result for human
        pred_class=np.argmax(preds, axis=-1)
        print(pred_class)
        #pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1])               # Convert to string
        if pred_class[0]== [1]:
            result = ("Preliminary diagnosis suggests Non-Covid presentation with Score:"+str(preds[0][0]))
        elif pred_class[0]== [0]:
            result = ("Preliminary diagnosis suggests Covid presentation with Score:"+str(preds[0][1]))
        return result
    return result


if __name__ == '__main__':
     app.run(port=3306, debug=True)
   
    # Serve the app with gevent
    #http_server = WSGIServer(('', 4006), app)
    #http_server = WSGIServer(('', int(os.environ.get('PORT'))), app)
     http_server = WSGIServer(('', int(os.environ.get('port'))), app)
     http_server.serve_forever()
