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

import numpy as np

import tensorflow as tf
import re

graph = tf.Graph()

class Prediction(Base):
  # @classmethod
  # def process(cls, sentence=None):

  #   return False
  @classmethod
  def process(self, sentence):
    print("Coming into prediction prcess")
    test_data = self.__getSentenceMatrix(sentence)

    tf.stack(np.asarray(test_data))
    with graph.as_default():
      weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
      bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))

      input_data = tf.placeholder(tf.float32, [batchSize, TrainingSet.maxSentenceLen(), 300], name = 'input_placeholder')
      labels = tf.placeholder(tf.float32, [batchSize, numClasses], name = 'labels_placeholder')

      global_step = tf.contrib.framework.get_or_create_global_step()

      lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
      lstmCell = tf.contrib.rnn.DropoutWrapper(cell = lstmCell, output_keep_prob=0.75)
      outputs, _ = tf.nn.dynamic_rnn(lstmCell, input_data, dtype=tf.float32)

      outputs = tf.transpose(outputs, [1, 0, 2])
      last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)
      prediction = (tf.matmul(last, weight) + bias)
      correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
      accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
      train_step = (tf.train.AdamOptimizer().minimize(loss, global_step = global_step))
        

      saver = tf.train.Saver()

    with tf.Session(graph = graph) as sess:

      ckpt = tf.train.get_checkpoint_state('logs')
        #print("-----------------", tf.global_variables())
      print("==============================================", ckpt)
      if ckpt and ckpt.model_checkpoint_path:

        print('Start load model')
        saver.restore(sess, ckpt.model_checkpoint_path)
      else:
        print('no checkpoint found')
        return
    # predictedSentiment = sess.run(prediction, {input_data:test_data})
    # # with tf.Session() as sess:
    # #   new_saver = tf.train.import_meta_graph('')
    # #   new_saver.restore(sess, tf.train.latest_checkpoint(''))
    # #   prediction = tf.get_collection('pred_network')[0]
    # #   preddictedSentiment = sess.run(prediction, {input_data: test_data})
    # print("Checking predictedSentiment ... ", predictedSentiment)
    # if (predictedSentiment[0])>(predictedSentiment[1]):
    #   print("Positive value is ", predictedSentiment[0])
    #   return 1
    # else:
    #   print("Negative value is ", predictedSentiment[1])
    #   return 0


  @classmethod
  def __getSentenceMatrix():
    print("++++++++++++++++", string)
    cleanedSentence = self.__cleanSentences(string)
    split = cleanedSentence.split()
    batch = []
    matrix = []
    for word in enumerate(split):
      if word:
        word_vector = em.where('word', word).first().word_vector
        matrix.append(word_vector)
      else:
        matrix.append([0] * 300)
    for l in range(len(matrix), ts.maxSentenceLen() - 1):
      matrix.append([0] * 300)
      batch.append(matrix)
      tf.stack(np.asarray(batch))

    return batch



  @classmethod
  def __createOP(self):
    weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    
    print("create input")
    input_data = tf.placeholder(tf.float32, [batchSize, TrainingSet.maxSentenceLen(), 300], name = 'input_placeholder')
    labels = tf.placeholder(tf.float32, [batchSize, numClasses], name = 'labels_placeholder')

    print("global step")
    global_step = tf.contrib.framework.get_or_create_global_step()

    print("initial lstm cell")
    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell = lstmCell, output_keep_prob=0.75)

    print("Done graph")
    outputs, _ = tf.nn.dynamic_rnn(lstmCell, input_data, dtype=tf.float32)
    outputs = tf.transpose(outputs, [1, 0, 2])
    last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)

    return prediction

  @classmethod
  def __cleanSentences(self, string):
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    string = re.sub(strip_special_chars, "", string.lower())
    return string




