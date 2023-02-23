from flask import Flask, request, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

import os

import cv2
import numpy
from PIL import Image

from .classification import classify
# def classify(*args):
#     return "None"

IGNORE_FILE_EXTENSION = True

UPLOAD_FOLDER = os.path.abspath('./uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
cors = CORS(app)

app.config['SECRET_KEY'] = 'secret shhh'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ROOT_DIR'] = os.path.dirname(os.path.realpath(__file__))
app.config['STATIC_DIR'] = os.path.join(app.config['ROOT_DIR'], 'static')
app.config['INDEX_PAGE_DIR'] = os.path.join(
    app.config['STATIC_DIR'], 'index.html')


def allowed_extension(filename: str):
    return '.' in filename and filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def route_index():
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def route_classify():
    files = list(request.files.values())
    if len(files) == 0:
        return ("No file was uploaded with POST request", 400)
    file = files[0]

    if file.filename == '':
        return ("No file selected in POST request", 400)

    if IGNORE_FILE_EXTENSION:
        pass  # ignore file extension checking
    elif not allowed_extension(file.filename):
        return ("Unallowed Filetype Detected", 400)

    if file:
        filename = secure_filename(file.filename)

        try:
            pil_image = Image.open(file.stream).convert('RGB')
            cv_image = numpy.array(pil_image)

            # Converting to grayscale
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)

        except Exception as e:
            return (f"Provided file is an invalid image ({e})", 400)

        try:
            return classify(cv_image)
        except Exception as e:
            return (f"Provided file could not be classified ({e})", 400)

    return ("Unknown Error occurred", 500)
