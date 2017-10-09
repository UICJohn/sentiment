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
import os

class Tester(Base):
  @classmethod
  def process(self):
    print('Call Tester process')
    # graph = tf.Graph()
    # with graph.as_default():

    #   weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
    #   bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))

    #   input_data = tf.placeholder(tf.float32, [batchSize, TrainingSet.maxSentenceLen(), 300], name = 'input_placeholder')
    #   labels = tf.placeholder(tf.float32, [batchSize, numClasses], name = 'labels_placeholder')

    #   global_step = tf.contrib.framework.get_or_create_global_step()

    #   lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    #   lstmCell = tf.contrib.rnn.DropoutWrapper(cell = lstmCell, output_keep_prob=0.75)
    #   outputs, _ = tf.nn.dynamic_rnn(lstmCell, input_data, dtype=tf.float32)

    #   outputs = tf.transpose(outputs, [1, 0, 2])
    #   last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

    #   prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
    #   correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
    #   accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
    #   loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    #   train_step = (tf.train.AdamOptimizer().minimize(loss, global_step = global_step))
    #   saver = tf.train.Saver()

    # with tf.Session(graph = graph) as sess:
    #   sess.run(tf.global_variables_initializer())
    #   ckpt = tf.train.get_checkpoint_state('logs')
    #   if ckpt and tf.gfile.Exists('logs'):
    #     saver.restore(sess, ckpt.model_checkpoint_path)
    #     sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels}) * 100
    self.__getTestMatrix()

  @classmethod
  def __getTestMatrix(self):
    print('Preparing test data')
    self.eachFile('/Users/chih/Documents/IOS/dev_sentiment/sentiment/neg')


  @classmethod
  def eachFile(self,filePath):
    pathDir = os.listdir(filePath)
    num = 0;
    for allDir in pathDir:
      num = num + 1
      child = os.path.join('%s%s'%(filePath, allDir))
      print('==========================', allDir)
      print('---------------', child)
      print('===========================', open(child))
      #print('The input txt is ', self.readFile(allDir))
    #print('===========================', open('0_2.txt'))
    print('file number is ', num)

  # @classmethod
  # def readFile(self,filename):
  #   fopen = open(filename, 'r')
  #   string = ''
  #   for eachLine in fopen:
  #     string = self.__cleanSentences(eachLine)
  #   fopen.close()
  #   return string







  @classmethod
  def __cleanSentences(self, string):
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    string = re.sub(strip_special_chars, "", string.lower())
    return string


