import pdb
from flask import Flask, request
from flask_restful import Resource, Api
from app.models import db_conn
from app.config import *
from app.api import TrainningSetController
from app.api import TrainningTaskController


app = Flask(__name__)

app.config.from_object(__name__)
api = Api(app)
db_conn.init_app(app)

api.add_resource(TrainningSetController, '/')
api.add_resource(TrainningTaskController, '/trainning')
if __name__ == '__main__':
    app.run()