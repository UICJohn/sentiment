from flask_restful import Resource,reqparse,request
from ..lib import Prediction
from ..lib import Classifier

class PredictionController(Resource):

	def post(self):
		parser = reqparse.RequestParser()
		parser.add_argument('sentence')
		args = parser.parse_args()

		if args['sentence'] and Prediction.process(args['sentence']):
			return {'status': 'DONE'}
		else:
			return 'ERROR', 404



  def get(self):
    #TODO predict input sentence
    predicted()