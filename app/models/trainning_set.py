from orator.orm import has_one
from .base import Base
class TrainningSet(Base):
	__casts__ = {'word_ids': 'array'}