#!/usr/bin/env python3
from flask import Flask
import os

UPLOAD_FOLDER = './app/uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.urandom(16)

from app import routes
