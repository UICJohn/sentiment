from .application import *
from .cache import *
from flask_orator import Orator, jsonify
from flask import Flask, request
from flask_restful import Resource, Api
from celery import Celery

def init_celery(app):
	celery = Celery('sentiment', broker = brokerUrl)
	celery.conf.update(app.config)
	TaskBase = celery.Task
	class ContextTask(TaskBase):
		abstract = True
		def __call__(self, *args, **kwargs):
			with app.app_context():
				return TaskBase.__call__(self, *args, **kwargs)
	celery.Task = ContextTask
	return celery

app = Flask(__name__)
app.config.from_object(__name__)
api = Api(app)
celery_app = init_celery(app)
db_conn = Orator(app)