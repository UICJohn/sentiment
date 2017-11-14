from ..models import TrainingSet, EmMatrix
from .base import Base
from app.config import redis, batchSize, maxBatchCount, max_epoch
import redis_lock
from flask import current_app
from .queue import Queue
import pickle
class Batch(Base):
  @classmethod
  def enqueue(cls):
    with current_app.test_request_context():
      q = Queue("batch")
      training_set_count = int(TrainingSet.count()/batchSize)
      for j in range(0, max_epoch):
        for i in range(0, training_set_count):
          training_sets = TrainingSet.select("id", "word_ids").paginate(batchSize, i)
          ids_arr = [training_set.id for training_set in training_sets]
          q.push(ids_arr)

  @classmethod
  def dequeue(cls):
     q = Queue("batch")
     batch = q.pop()
     if(batch):
      return batch
     else:
       return None

  @classmethod
  def can_batch(cls, queue):
    batch_count = queue.size()
    if(not batch_count):
      return True
    elif(int(batch_count) < maxBatchCount):
      return True
    else:
      return False

  @classmethod
  def steps_per_epoch(cls):
    return int(TrainingSet.count()/batchSize)