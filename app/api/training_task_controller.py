from flask_restful import Resource
from sentiment_task import train
from ..lib import Trainer
from ..config import batchSize
class TrainingTaskController(Resource):
	def get(self):
		task_info = {
			"word": "hello world"
		}
		return task_info

	def post(self):
		# train.delay(batchSize)
		label = Trainer(batchSize)
		return {"STATUS": label}