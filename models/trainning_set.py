from orator import Model
from app import db
from orator.orm import has_one

class TrainningSet(db.Model):
	@has_one('foreign_key', 'set_id')
	def trainning_vector(self):
		return TrainningVector