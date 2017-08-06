from flask_restful import Resource
from ..models import TrainningSet
from ..models import db_conn
from ..config import batch_size

class Train(Resource):
  def get(self):
  	set_info = {
  		'Count': TrainningSet.count(),
  		'Maximum Word Length': TrainningSet.select(db_conn.raw("max(array_length(regexp_split_to_array(words, '\s | \t'), 1)) as max_len")).first().max_len,
  		'Batch Size': batch_size,
  		'Batch Count': 2
  	}
  	return set_info