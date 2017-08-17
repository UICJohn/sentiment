from .config import database
from flask_orator import Orator, jsonify
from flask import Flask, request
from flask_restful import Resource, Api
from celery import Celery

def init_celery(app):
	celery = Celery(app.import_name, broker=app.config['CELERY_BROKER_URL'])
	celery.conf.update(app.config)
	TaskBase = celery.Task
	class ContextTask(TaskBase):
		abstract = True
		def __call__(self, *args, **kwargs):
			with app.app_context():
				return TaskBase.__call__(self, *args, **kwargs)
	celery.Task = ContextTask
	return celery

ORATOR_DATABASES = database
sentiment_app = Flask(__name__)
sentiment_app.config["CELERY_BROKER_URL"] = 'redis://192.168.0.6:6379'
sentiment_app.config.from_object(__name__)
sentiment_api = Api(sentiment_app)
db_conn = Orator(sentiment_app)