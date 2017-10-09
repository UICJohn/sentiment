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
import datetime
import numpy as np

import tensorflow as tf
import re

#tf.reset_default_graph()

class Prediction(Base):

  @classmethod
  def process(self, sentence):
    # print("Coming into prediction prcess")
    test_data = self.__getSentenceMatrix(sentence)
    graph = tf.Graph()

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

        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
        correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
        accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
        train_step = (tf.train.AdamOptimizer().minimize(loss, global_step = global_step))
        saver = tf.train.Saver()

    prediction_result  = []
    final_result = ''
    with tf.Session(graph = graph) as sess:
        ckpt = tf.train.get_checkpoint_state('logs')
        if ckpt and tf.gfile.Exists('logs'):
            print('Start load model')
            saver.restore(sess, ckpt.model_checkpoint_path)
            predictedSentiment = sess.run(prediction, {input_data:test_data})[0]
            print('====================================', predictedSentiment)
            if (predictedSentiment[0])>(predictedSentiment[2]):
                print("Positive value is ", predictedSentiment[0])
                final_result = 'Positive'
                prediction_result.append(predictedSentiment)
                prediction_result.append('Positive')
            else:
                prediction_result.append(predictedSentiment)
                prediction_result.append('Negative')
                final_result = 'Negative'

        else:
            print('no checkpoint found')
            return
    return prediction_result

  @classmethod
  def __getSentenceMatrix(self, string):
    cleanedSentence = self.__cleanSentences(string)
    print('-----------------', cleanedSentence)
    split = cleanedSentence.split()
    batch = []
    final = []
    for word in split:
      matrix = []
      if word:
        if em.where('word', word).first():
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
    return batch

  @classmethod
  def __cleanSentences(self, string):
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    string = re.sub(strip_special_chars, "", string.lower())
    return string