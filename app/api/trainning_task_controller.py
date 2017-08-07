from flask_restful import Resource
from sentiment_task import train

class TrainningTaskController(Resource):
	def get(self):
		task_info = {
			"word": "hello world" 
		}
		return task_info

	def post(self):
		train.delay()
		return {"STATUS": "REQUEST HAS BEEN ADD TO QUEUE"}