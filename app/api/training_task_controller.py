from flask_restful import Resource
from ..lib import Batch
from ..config import batchSize

class TrainingTaskController(Resource):
	def get(self):
		task_info = {
			"word": "hello world"
		}
		return task_info

	def post(self):
		# create_batch.delay(1)
		Batch.enqueue()
		return {"STATUS": "DONE"}