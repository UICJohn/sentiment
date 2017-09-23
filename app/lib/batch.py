from ..models import TrainingSet, EmMatrix
from .base import Base
from app.config import redis, batchSize, maxBatchCount
import redis_lock
from flask import current_app
from .queue import Queue

class Batch(Base):
  @classmethod
  def get_batch(cls):
    with redis_lock.Lock(redis, "batch_lock", expire = 60):
      maxSentenceLen = TrainingSet.maxSentenceLen()
      training_sets = TrainingSet.where("iterations", cls.current_epoch()).order_by_raw("random()").paginate(batchSize, 1)
      for i in range(0, len(training_sets)):
        training_sets[i].iterations += 1
        training_sets[i].save()
      batch = cls.__vector2matrix(training_sets, maxSentenceLen)
    return batch

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
  def current_epoch(cls):
    epoch = redis.get("current_epoch")
    if(epoch):
      return int(epoch)
    else:
      redis.incr("current_epoch")
      return 1

  @classmethod
  def __vector2matrix(cls, training_sets, max_sentence_len):
    labels = []
    data = []
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
      data.append(matrix)
      labels.append(label)
    return [data, labels]