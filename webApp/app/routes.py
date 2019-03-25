#!/usr/bin/env python3
from app import app
import os
from flask import render_template, flash, request, redirect, url_for, Response
from flask import session
from werkzeug.utils import secure_filename
from app.wrangle_data import classify_dog, add_face_rectangles
import cv2
import base64
import io

# http://flask.pocoo.org/docs/1.0/patterns/fileuploads/
ALLOWED_EXTENSIONS = set(['jpeg', 'jpg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part in posted form')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an ampty part without fileName
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            jpegdata = file.read()
            image_feed = 'data:image/jpeg;base64,' + base64.b64encode(jpegdata).decode('ascii')
            image_path = io.BytesIO(jpegdata)  # we can use jpeg data on memory as a file

            result_str, prediction = classify_dog(image_path)

            if result_str == 'face':
                message = 'Detected a face looks like '
            elif result_str == 'dog':
                message = 'Detected a dog looks like '
            else:
                message = 'Neither dog nor face'

            if prediction:
                if result_str == 'face':
                    img = add_face_rectangles(image_path)
                    jpegdata = cv2.imencode('.jpg', img)[1]
                    image_feed = 'data:image/jpeg;base64,' + base64.b64encode(jpegdata).decode('ascii')

                ref_url = 'https://www.google.com/search?q=dog+breed+' + prediction.replace(' ', '+') + '&tbm=isch'
            else:
                ref_url = ''
            print(ref_url)

            return render_template('index.html', message=message,
                        prediction=prediction, ref_url=ref_url, image_feed=image_feed)

    return render_template('index.html', message=None, prediction=None, ref_url=None,image_feed=None)
