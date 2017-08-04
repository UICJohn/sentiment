from flask_restful import Resource
from ..models import TrainningSet

class Train(Resource):
  def get(self):
  	set_info = {
  		'Count': TrainningSet.count(),
  		'Maximum Length': TrainningSet
  	}
  	return set_info