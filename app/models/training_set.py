from .base import Base
class TrainingSet(Base):
	__casts__ = {
    'word_ids': 'list'
	}