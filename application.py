import pdb
from flask import Flask, request
from flask_restful import Resource, Api
from app.models import db_conn
from app.config import *
from app.api import TrainingSetController
from app.api import TrainingTaskController

app = Flask(__name__)

app.config.from_object(__name__)
api = Api(app)
db_conn.init_app(app)

api.add_resource(TrainingSetController, '/')
api.add_resource(TrainingTaskController, '/training')
if __name__ == '__main__':
    app.run()