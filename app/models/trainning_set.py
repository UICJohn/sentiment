from orator import Model
from orator.orm import has_one
# from application import db

class TrainningSet(db.Model):
	@has_one('foreign_key', 'set_id')
	def trainning_vector(self):
		return TrainningVector