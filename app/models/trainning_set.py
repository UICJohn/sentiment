from orator.orm import has_one
from .base import Base
class TrainningSet(Base):
	@has_one('foreign_key', 'set_id')
	def trainning_vector(self):
		return TrainningVector