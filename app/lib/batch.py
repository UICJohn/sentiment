from ..models import TrainingSet, db_conn, EmMatrix
from .base import Base
from app.config import redis, batchSize, maxBatchCount
import pdb, json

class Batch(Base):

  @classmethod
  def enqueue(cls, quantity = 1 ):
    maxVectorSize = TrainingSet.maxVectorSize()
    setsCount = TrainingSet.where("trained", False).count()
    for i in range(0, quantity):
      training_sets = TrainingSet.where("trained", False).order_by_raw("random()").paginate(batchSize, i)
      batch = cls.__vector2matrix(training_sets, maxVectorSize)
      if cls.can_batch():
        batch_index = cls.__current_batch()
        redis.forever(str(batch_index), json.dumps(batch))
        pdb.set_trace()
        redis.increment('batch_count')

  @classmethod
  def dequeue(cls):
    batch_index = cls.__current_batch()
    batch = redis.get(str(batch_index))
    redis.remove(str(batch_index))
    redis.decrement('batch_count')
    return json.loads(batch)

  @classmethod
  def can_batch(cls):
    return True if (redis.get("batch_count") < maxBatchCount) else False

  @classmethod
  def __current_batch(cls):
    current_batch = redis.get("current_batch")
    if current_batch:
      return current_batch
    else:
      for i in range(0, maxBatchCount):
        if redis.get("batch-"+str(i)):
          redis.forever("current_batch", i)
          return "batch-"+str(i)
    return "batch-"+str(0)

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