from ..models import TrainingSet, EmMatrix
from .. import db_conn
from .base import Base
from app.config import redis, batchSize, maxBatchCount
from flask import current_app
import pdb, json
from .queue import Queue

class Batch(Base):

  @classmethod
  def enqueue(cls, quantity = 1 ):
    with current_app.test_request_context():
      q = Queue("batch")
      maxSentenceLen = TrainingSet.maxSentenceLen()
      setsCount = TrainingSet.where("trained", False).count()
      for i in range(0, quantity):
        training_sets = TrainingSet.where("trained", False).order_by_raw("random()").paginate(batchSize, i)
        batch = cls.__vector2matrix(training_sets, maxSentenceLen)
        if cls.can_batch(q):
          q.push(batch)
          redis.incr('batch_count')

  @classmethod
  def dequeue(cls):
    q = Queue("batch")
    batch_index = cls.__current_batch()
    batch = q.pop()
    if(batch):
      redis.decr('batch_count')
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
  def __current_batch(cls):
    current_batch = redis.get("current_batch")
    if current_batch:
      return current_batch
    else:
      for i in range(0, maxBatchCount):
        if redis.get("batch-"+str(i)):
          redis.set("current_batch", i)
          return "batch-"+str(i)
    return "batch-"+str(0)

  @classmethod
  def __vector2matrix(cls, training_sets, max_sentence_len):
    labels = []
    batch = []
    for training_set in training_sets:
      matrix = []
      label = [0] * 3
      word_ids = training_set.word_ids
      for word_id in word_ids:
        word = EmMatrix.where('id', word_id).first()
        if word:
          matrix.append(word.vector)
        else:
          matrix.append([0] * 300)
      for l in range(len(matrix), max_sentence_len):
        matrix.append([0] * 300)
        label[training_set.label + 1] = 1
      batch.append(matrix)
      labels.append(label)
    return [batch, labels]