from flask_restful import Resource
from ..models import TrainingSet
from ..models import db_conn
from ..config import batchSize, redis
class TrainingSetController(Resource):
  def get(self):
    if redis.has('max_word_length'):
      redis.get('max_word_length')
    else:
      TrainingSet.select(db_conn.raw("max(array_length(regexp_split_to_array(words, '\s|\t'), 1)) as max_len")).first().max_len
      redis.put('max_word_length', max_word_length, 180)
    set_info = {'Count': TrainingSet.count(), 'Maximum Word Length': redis.get('max_word_length'), 'Batch Size': batchSize, 'Batch Count': 2}
    return set_info

  def post(self):
  	# TODO add trainning set
  	pass