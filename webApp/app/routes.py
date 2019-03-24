#!/usr/bin/env python3
from app import app
import os
from flask import render_template, flash, request, redirect, url_for, Response
from flask import session
from werkzeug.utils import secure_filename
from app.wrangle_data import classify_dog, add_face_rectangles
import cv2

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
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)
            result_str, prediction = classify_dog(image_path)
            session['image_path'] = image_path
            session['result_str'] = result_str
            if result_str == 'face':
                result_str = 'Detected a face looks like '
            elif result_str == 'dog':
                result_str = 'Detected a dog looks like '
            else:
                result_str = 'Neither dog nor face'

            if prediction:
                ref_url = 'https://www.google.com/search?q=dog+breed+' + prediction.replace(' ', '+') + '&tbm=isch'
            else:
                ref_url = ''
            print(ref_url)

            return render_template('index.html', result_str=result_str,
                        prediction=prediction, ref_url=ref_url)

    return render_template('index.html', result_str=None, prediction=None, ref_url=None)

@app.route('/image_feed')
def image_feed():
    """Image data feed. Put this in the src attribute of an img tag."""
    if 'image_path' in session:
        img = cv2.imread(session['image_path'])

        if 'result_str' in session and session['result_str'] == 'face':
            img = add_face_rectangles(session['image_path'])

        frame = cv2.imencode('.jpg', img)[1].tobytes()
        os.remove(session['image_path'])
        return Response(b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n',
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response("",
                    mimetype='multipart/x-mixed-replace; boundary=frame')
