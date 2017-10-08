import pickle
from flask import current_app
from ..config import vectors_redis
from ..models import TrainingSet, EmMatrix
class Vector:
  def add(self, training_set):
    for word_id in training_set.word_ids:
      vectors_redis.hsetnx("vectors", word_id, pickle.dumps(EmMatrix.find_or_fail(word_id).vector))

  def fetch(self, key):
    return pickle.loads(vectors_redis.hget("vectors", key))

  def process_all(self):
    with current_app.test_request_context():
      counter = 0
      for training_set in TrainingSet.get():
        self.add(training_set)
        print("TrainingSet Vector Count: %d" % counter)
        counter += 1
