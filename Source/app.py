from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

#bibliotecas Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

#bibliotecas Flask
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.wsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Carrega o modelo keras salvo
MODEL_PATH = 'models/basicCNN.h5'
model = load_model(MODEL_PATH)
model._make_predict_function() # Necessário para funcionar
print('Modelo carregado')

#Carrega as labels utilizadas pelo Modelo
label_dict = dict()
with open('static/files/list_category_cloth.txt') as f:
    lineno = 1
    for line in f:
        arglist = ' '.join(line.split()).split(' ')
        label_dict.update({lineno:arglist[0]})
        lineno += 1


# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(300, 200))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

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

        # Process your result for human
        pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = label_dict.get(int(pred_class))             # Convert to string
        return result
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()