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
          print("TrainingSet Count: %d" % i)
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
  # def __vector2matrix(cls, training_sets, max_sentence_len):
  #   labels = []
  #   data = []
  #   for training_set in training_sets:
  #     matrix = []
  #     label = [0] * 3
  #     word_ids = training_set.word_ids
  #     for word_id in word_ids:
  #       word = EmMatrix.where('id', word_id).first()
  #       if word:
  #         matrix.append(word.vector)
  #       else:
  #         matrix.append([0] * 300)
  #     for l in range(len(matrix), max_sentence_len):
  #       matrix.append([0] * 300)
  #       label[training_set.label + 1] = 1
  #     data.append(matrix)
  #     labels.append(label)
  #   return [data, labels]
