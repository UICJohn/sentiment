import os
from flask import Flask, request
from flask_orator import Orator, jsonify
from config import database

DEBUG = True
ORATOR_DATABASES = database

app = Flask(__name__)
app.config.from_object(__name__)

db = Orator(app)

if __name__ == '__main__':
    app.run()