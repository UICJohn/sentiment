from ..models import TrainingSet, db_conn, EmMatrix
from .base import Base
import pdb

class Trainer(Base):

	def __init__(self, batchSize):
		self.preProcess(batchSize)

	def preProcess(self, batchSize):
		# TODO replace maxvector with cache
		# maxVector = TrainingSet.select(db_conn.raw("max(array_length(regexp_split_to_array(words, '\s|\t'), 1)) as max_len")).first().max_len
		setsCount = TrainingSet.where("trained", False).count()
		for batch in range(0, setsCount/batchSize):
			training_sets = TrainingSet.where("trained", False).paginate(batchSize, batch)
			for training_set in training_sets:
				pdb.set_trace()