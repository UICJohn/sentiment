from ..models import EmMatrix as em
from ..models import TrainingSet as ts
from ..models import TrainingSet, EmMatrix
from .base import Base
from flask_restful import Resource
from ..models import TrainingSet
from ..config import batchSize, redis
from .. import db_conn
from ..config import batchSize, numClasses, lstmUnits, cluster_spec, max_epoch
from .batch import Batch
from os import listdir
from os.path import isfile, join
import datetime
import numpy as np

import tensorflow as tf
import re
import os

class Tester(Base):
  @classmethod
  def process(self):
    print('Call Tester process')
    graph = tf.Graph()
    with graph.as_default():

      weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
      bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))

      input_data = tf.placeholder(tf.float32, [batchSize, TrainingSet.maxSentenceLen(), 300], name = 'input_placeholder')
      labels = tf.placeholder(tf.float32, [batchSize, numClasses], name = 'labels_placeholder')
      print('++++++++++++++++++++++ graph', labels)
      global_step = tf.contrib.framework.get_or_create_global_step()

      lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
      lstmCell = tf.contrib.rnn.DropoutWrapper(cell = lstmCell, output_keep_prob=0.75)
      outputs, _ = tf.nn.dynamic_rnn(lstmCell, input_data, dtype=tf.float32)

      outputs = tf.transpose(outputs, [1, 0, 2])
      last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

      prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
      correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
      accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
      train_step = (tf.train.AdamOptimizer().minimize(loss, global_step = global_step))
      saver = tf.train.Saver()

    test_data, labels_temp = self.__getTestMatrix()
    np.save("test_matrix.npy",test_data)
    
    with tf.Session(graph = graph) as sess:
      sess.run(tf.global_variables_initializer())
      ckpt = tf.train.get_checkpoint_state('logs')
      if ckpt and tf.gfile.Exists('logs'):
        saver.restore(sess, ckpt.model_checkpoint_path)
        a = np.load("test_matrix.npy")
        #print('~~~~~~~~~~~~~~~~~', labels)
        for i in range(1,25):
          print("Accuracy for this batch:", (sess.run(accuracy, {input_data: a[0], labels : labels_temp})) * 100 )
        # sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels}) * 100
      else:
        print('No checkpoint file')
    # np.save("test_matrix.npy",test_data)



  @classmethod
  def __getTestMatrix(self):
    print('Preparing test data')
    test_data, labels = self.eachFile('/Users/chih/Documents/IOS/dev_sentiment/sentiment/neg/')
    print('=========', len(test_data))
    return test_data,labels

  @classmethod
  def eachFile(self,filePath):
    num = 0
    test_data = []
    batch = []
    files = [filePath + f for f in listdir(filePath) if isfile(join(filePath, f))]
    labels = []
    for f in files:
      num = num + 1
      with open(f, "r", encoding = 'utf-8') as f:
        line = f.readline()
        line = self.__cleanSentences(line)
        split = line.split()
          # print('-----------------------', split)
        for word in split:
          matrix = []
          if word:
            if em.where('word',word).first():
                # print('+++++++++++++++++',word)
              word_vector = em.where('word', word).first().vector
            else:
              word_vector.append([0] * 300)
            matrix.append(word_vector)
          else:
            matrix.append([0] * 300)
        for l in range(len(matrix), TrainingSet.maxSentenceLen()):
          matrix.append([0] * 300)
        for i in range(1,batchSize+1):
          batch.append(matrix)
          labels.append([0,0,1])
        test_data.append(batch)
        print('The num is', num)
    return test_data, labels
    

  @classmethod
  def __cleanSentences(self, string):
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    string = re.sub(strip_special_chars, "", string.lower())
    return string


