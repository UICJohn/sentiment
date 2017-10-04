from ..models import TrainingSet, EmMatrix
from datetime import datetime
import json
class Base():
  def vector2matrix(self, training_set_ids):
    labels = []
    data = []
    max_sentence_len = TrainingSet.maxSentenceLen()
    training_sets = TrainingSet.select("word_ids", "label").where_in("id", training_set_ids).get()
    for training_set in training_sets:
      print("start %s" % str(datetime.now()))
      label = [0] * 3
      word_ids = training_set.word_ids
      words = EmMatrix.select("vector").where_in('id', word_ids).get()
      matrix = [ word.vector for word in words ]
      print("Matrix Size: %d : Appending 0 matrix %s" % (len(matrix), str(datetime.now())))
      for l in range(len(matrix), max_sentence_len):
        matrix.append([0] * 300)
      label[training_set.label + 1] = 1
      data.append(matrix)
      labels.append(label)
      print("end %s \n" % str(datetime.now()))
    return [data, labels]
