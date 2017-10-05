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

#tf.reset_default_graph()

class Prediction(Base):

  @classmethod
  def process(self, sentence):
    print("Coming into prediction prcess")
    print('Input sentence is ' ,sentence)
    test_data = self.__getSentenceMatrix(sentence)
    # print('======================', len(test_data[0]))
    # print('\n')
    print('test data is ', len(test_data))
    print('test data details ', len(test_data[0]))
    # print('=================',len(self.__getSentenceMatrix(sentence)))
    test_data = tf.stack(np.asarray(test_data))
    print('======================',test_data)
    tf.reshape(test_data, [batchSize, TrainingSet.maxSentenceLen(), 300])
    print('reshape ======================',test_data)
    print('======================', tf.__version__)

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
        prediction = (tf.matmul(last, weight) + bias)
        correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
        accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
        train_step = (tf.train.AdamOptimizer().minimize(loss, global_step = global_step))
        saver = tf.train.Saver()

    with tf.Session(graph = graph) as sess:
        ckpt = tf.train.get_checkpoint_state('logs')
        print("==============================================", ckpt)
        print('\n')
        print("==============================================", tf.gfile.Exists('logs'))
        if ckpt and tf.gfile.Exists('logs'):
            print('Start load model')
            saver.restore(sess, ckpt.model_checkpoint_path)
            predictedSentiment = sess.run(prediction, {input_data:test_data})[0]
            print("Checking predictedSentiment ... ", predictedSentiment)
            if (predictedSentiment[0])>(predictedSentiment[1]):
                print("Positive value is ", predictedSentiment[0])
            else:
                print("Negative value is ", predictedSentiment[1])

        else:
            print('no checkpoint found')
            return

  @classmethod
  def __getSentenceMatrix(self, string):
    cleanedSentence = self.__cleanSentences(string)
    print('-----------------', cleanedSentence)
    split = cleanedSentence.split()
    batch = []
    print('the split is ', split)
    for word in split:
      matrix = []
      if word:
        word_vector = em.where('word', word).first().vector
        print('each work corresponding vector ----', word, word_vector)
        print('\n')
        #print('each work of sentence corresponding vector is ', len(word_vector))
        matrix.append(word_vector)
        #print('The shape is ', len(matrix))
      else:
        matrix.append([0] * 300)
    print('The shape of matrix is ', len(matrix))
    for l in range(len(matrix), TrainingSet.maxSentenceLen()):
      matrix.append([0] * 300)
    batch.append(matrix)
    # tf.convert_to_tensor(np.asarray(batch),dtype=np.float32)
    #tf.stack(np.asarray(batch))
    # print("the input batch shape is ", batch.shape)
    #print('getSentenceMatrix result is ', )
    return batch

  # @classmethod
  # def __createOP(self):


  #   weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
  #   bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    
  #   print("create input")
  #   input_data = tf.placeholder(tf.float32, [batchSize, TrainingSet.maxSentenceLen(), 300], name = 'input_placeholder')
  #   labels = tf.placeholder(tf.float32, [batchSize, numClasses], name = 'labels_placeholder')

  #   print("global step")
  #   global_step = tf.contrib.framework.get_or_create_global_step()

  #   print("initial lstm cell")
  #   lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
  #   lstmCell = tf.contrib.rnn.DropoutWrapper(cell = lstmCell, output_keep_prob=0.75)

  #   print("Done graph")
  #   outputs, _ = tf.nn.dynamic_rnn(lstmCell, input_data, dtype=tf.float32)
  #   outputs = tf.transpose(outputs, [1, 0, 2])
  #   last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)
  #   prediction = (tf.matmul(last, weight) + bias)

  #   return prediction

  @classmethod
  def __cleanSentences(self, string):
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    string = re.sub(strip_special_chars, "", string.lower())
    return string