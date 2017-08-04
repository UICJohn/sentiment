from flask import Flask, request
from flask_orator import Orator, jsonify
from flask_restful import Resource, Api
from app.config import database
# from app.api import Train
DEBUG = True
ORATOR_DATABASES = database

app = Flask(__name__)
app.config.from_object(__name__)
api = Api(app)

db = Orator(app)

# api.add_resource(Train, '/')
if __name__ == '__main__':
    app.run()