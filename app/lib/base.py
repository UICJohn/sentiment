from ..models import TrainingSet, EmMatrix
from datetime import datetime
from app.config import redis
import numpy as np
import pickle
from .vector import Vector
class Base():
  def vector2matrix(self, training_set_ids):
    v = Vector()
    start_at = datetime.now()
    labels = []
    data = []
    max_sentence_len = TrainingSet.maxSentenceLen()
    training_sets = TrainingSet.select("word_ids", "label").where_in("id", training_set_ids).get()
    for training_set in training_sets:
      label = [0] * 3
      word_ids = training_set.word_ids
      matrix = [ v.fetch(word_id) for word_id in word_ids ]
      matrix.extend([[0]*300] * (max_sentence_len - len(word_ids)))
      label[training_set.label + 1] = 1
      data.append(matrix)
      labels.append(label)
    end_at = datetime.now()
    print("RUN TIME: %dms" % int((end_at-start_at).total_seconds() * 1000.0) )
    return [data, labels]