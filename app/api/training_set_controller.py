from flask_restful import Resource
from ..models import TrainingSet
from ..models import db_conn
from ..config import batchSize
class TrainingSetController(Resource):
  def get(self):
  	set_info = {
  		'Count': TrainingSet.count(),
  		'Maximum Word Length': TrainingSet.select(db_conn.raw("max(array_length(regexp_split_to_array(words, '\s|\t'), 1)) as max_len")).first().max_len,
  		'Batch Size': batchSize,
  		'Batch Count': 2
  	}
  	return set_info

  def post(self):
  	# TODO add trainning set
  	pass