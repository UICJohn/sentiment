import pdb
from flask import Flask, request
from flask_restful import Resource, Api
from app.models import db_conn
from app.config import database
from app.api import Train



app = Flask(__name__)
ORATOR_DATABASES = database
DEBUG = True
app.config.from_object(__name__)
api = Api(app)
db_conn.init_app(app)

api.add_resource(Train, '/')
if __name__ == '__main__':
    app.run()