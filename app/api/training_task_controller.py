from flask_restful import Resource
from sentiment_task import create_batch
from ..lib import Batch
from ..config import batchSize
class TrainingTaskController(Resource):
	def get(self):
		task_info = {
			"word": "hello world"
		}
		return task_info

	def post(self):
		# train.delay(batchSize)
		Batch.store()
		return {"STATUS": "DONE"}