from .config import database
from flask_orator import Orator, jsonify
from flask import Flask, request
from flask_restful import Resource, Api

ORATOR_DATABASES = database

app = Flask(__name__)
app.config.from_object(__name__)
sentiment_api = Api(app)
db_conn = Orator(app)