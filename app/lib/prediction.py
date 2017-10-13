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
    print("Coming into prediction prcess")
    test_data = self.__getSentenceMatrix(sentence)
    #print("-=====================", test_data.shape)
    graph = tf.Graph()

    with graph.as_default():
        weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
        bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))

        input_data = tf.placeholder(tf.float32, [batchSize, TrainingSet.maxSentenceLen(), 300], name = 'input_placeholder')
        labels = tf.placeholder(tf.float32, [batchSize, numClasses], name = 'labels_placeholder')

        global_step = tf.contrib.framework.get_or_create_global_step()

        lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
        lstmCell = tf.contrib.rnn.DropoutWrapper(cell = lstmCell, output_keep_prob= 1)
        outputs, _ = tf.nn.dynamic_rnn(lstmCell, input_data, dtype=tf.float32)

        outputs = tf.transpose(outputs, [1, 0, 2])
        last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

        prediction = tf.matmul(last, weight) + bias
        correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
        accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
        train_step = (tf.train.AdamOptimizer().minimize(loss, global_step = global_step))
        saver = tf.train.Saver()

    prediction_result  = []
    final_result = ''

    temp_label = []
    for i in range(0,24):
        a = [1,0,0]
        temp_label.append(a)

    with tf.Session(graph = graph) as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state('logs')
        probability_value = 0
        if ckpt and tf.gfile.Exists('logs'):
            print('Start load model')
            saver.restore(sess, ckpt.model_checkpoint_path)

            print('------------------ accuracy', sess.run(last, {input_data:test_data}))
            predictedSentiment = sess.run(tf.nn.softmax(prediction), {input_data:test_data})[0]
            print('====================================', sess.run(prediction, {input_data:test_data}))
            if predictedSentiment[0]> predictedSentiment[2]:
                print("Positive value is ", predictedSentiment[0])
                final_result = 'Positive'
                #prediction_result.append(predictedSentiment)
                probability_value = predictedSentiment[0]
                #prediction_result.append('Positive')
            else:
                print("Negative value is ", predictedSentiment[2])
                #prediction_result.append(predictedSentiment)
                #print("Positive value is ", predictedSentiment[2])
                probability_value = predictedSentiment[2]
                #prediction_result.append('Negative')
                final_result = 'Negative'

        else:
            print('no checkpoint found') 
            return

    return final_result,probability_value 
  
  @classmethod
  def __getSentenceMatrix(self, string):
    cleanedSentence = self.__cleanSentences(string)
    print('-----------------', cleanedSentence)
    split = cleanedSentence.split()
    print('-----------------', split)
    batch = []
    final = []
    matrix = []
    word_vector = []
    for word in split:
        print('Coming in word in split')
        print(word)
        if word:
            print('Coming in if word:')
            if em.where('word', word).first():
                print('Could find word in embedding matrix')
                word_vector = em.where('word', word).first().vector

            else:
                print('Could not find word in embedding matrix')
                word_vector.append([0] * 300)     
        else:
            print('No word Null')
    matrix.append(word_vector)
    if len(split) >= TrainingSet.maxSentenceLen():
        print('The lenghth of input text >>>>>>>>>>> Max Sentence length')
        matrix = matrix[0:TrainingSet.maxSentenceLen()]
        for i in range(1,batchSize+1):
            batch.append(matrix)
    else:
        print('The lenghth of input text <<<<<<<<<<< Max Sentence length')
        for l in range(len(matrix), TrainingSet.maxSentenceLen()):
            matrix.append([0] * 300)
        for i in range(1,batchSize+1):
            batch.append(matrix)
    return batch
  # @classmethod
  # def __getSentenceMatrix(self, string):
  #   cleanedSentence = self.__cleanSentences(string)
  #   print('-----------------', cleanedSentence)
  #   split = cleanedSentence.split()
  #   print('-----------------', split)
  #   batch = []
  #   final = []
  #   for word in split:
  #     matrix = []
  #     print('==========',word)
  #     if word:
  #       if em.where('word', word).first():
  #           print('==========',word)
  #           word_vector = em.where('word', word).first().vector
  #       else:
  #           word_vector.append([0] * 300)
  #       matrix.append(word_vector)
  #     else:
  #       matrix.append([0] * 300)
  #   print('++++++++++++++++++++++++++', len(matrix[0]))
  #   for l in range(len(matrix), TrainingSet.maxSentenceLen()):
  #     matrix.append([0] * 300)
  #   for i in range(1,batchSize+1):
  #       batch.append(matrix)
  #   return batch

  @classmethod
  def __cleanSentences(self, string):
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    string = re.sub(strip_special_chars, "", string.lower())
    return string