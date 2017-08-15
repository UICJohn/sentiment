from ..models import TrainingSet, db_conn, EmMatrix
from .base import Base
from app.config import redis, batchSize
import pdb

class Batch(Base):

  @classmethod
  def store(cls, quantity = 1 ):
    maxVectorSize = TrainingSet.maxVectorSize()
    setsCount = TrainingSet.where("trained", False).count()
    for i in range(0, quantity):
      training_sets = TrainingSet.where("trained", False).order_by_raw("random()").paginate(batchSize, i)
      batch = cls.__vector2matrix(training_sets, maxVectorSize)
      if cls.can_batch():
        #TODO store batch to redis
        pdb.set_trace()
        redis.increment('batch_count', 1)

  @classmethod
  def fetch(cls):
    pass

  @classmethod
  def can_batch(cls):
    return True if (redis.get("batch_count") < 200) else False

  @classmethod
  def __vector2matrix(cls, vectors, max_vector_size):
    batch = []
    for training_set in vectors:
      matrix = []
      word_ids = training_set.word_ids
      for word_id in word_ids:
        #TODO dule with word cannot be found in em_matrix
        word = EmMatrix.where('id', word_id).first()
        if word:
          matrix.append(word.vector)
        else:
          matrix.append([0]*300)
      batch.append(matrix)
    return batch