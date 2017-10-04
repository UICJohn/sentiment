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

    print('======================', tf.__version__)
    # with tf.Session() as sess:
    #   #tf.initialize_all_variables().run()
    #   print('Start import meta graph')
    #   new_saver = tf.train.import_meta_graph('pretrained_lstm.ckpt-90000.meta')
    #   print('---------------------------')
    #   new_saver.restore(sess, tf.train.latest_checkpoint('model/'))
    #   print('Done import meta graph')
    #   prediction = tf.get_collection('pred_network')[0]
    #   preddictedSentiment = sess.run(prediction, {input_data: test_data}) 
    #
    #prediction = self.__createOP()
    

    # graph = tf.Graph()
    # with graph.as_default():
      
    #   saver = tf.train.Saver()

    # print('Start to load model and graph')
    
    # with tf.Session(graph = graph) as sess:
    #   #saver = tf.train.import_meta_graph('pretrained_lstm.ckpt-90000')
    #   saver.restore(sess, tf.train.latest_checkpoint('model'))


    # sess = tf.InteractiveSession()
    # #sess.run(init)
    # saver = tf.train.Saver()
    # saver.restore(sess, tf.train.latest_checkpoint('model'))


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


    # weight = tf.get_variable("weight", [1], tf.float32, initializer=tf.random_normal_initializer())
    # biase  = tf.get_variable("biase", [1], tf.float32, initializer=tf.random_normal_initializer())
    # print("create graph")
    # weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
    # bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
      
    # print("create input")
    # #Input
    # input_data = tf.placeholder(tf.float32, [batchSize, TrainingSet.maxSentenceLen(), 300], name = 'input_placeholder')
    # labels = tf.placeholder(tf.float32, [batchSize, numClasses], name = 'labels_placeholder')

    # print("global step")
    # #global_step
    # global_step = tf.contrib.framework.get_or_create_global_step()

    # print("initial lstm cell")
    # #initial lstm cell
    # lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    # lstmCell = tf.contrib.rnn.DropoutWrapper(cell = lstmCell, output_keep_prob=0.75)

    # print("Done graph")
    # #finalize graph
    # outputs, _ = tf.nn.dynamic_rnn(lstmCell, input_data, dtype=tf.float32)

    # print("loss and accuracy")
    # #define loss and accuracy
    # outputs = tf.transpose(outputs, [1, 0, 2])
    # last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)
    # prediction = (tf.matmul(last, weight) + bias)

    # saver = tf.train.Saver()
    # session = tf.Session()
    # ckpt = tf.train.get_checkpoint_state('logs')
    # print("==============================================", ckpt)
  
    # if ckpt and ckpt.model_checkpoint_path:
    #   print('Start load model')
    #   saver.restore(session, ckpt.model_checkpoint_path)
    #   global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
    #   print("%s, global_step = %d" % (ckpt.model_checkpoint_path, global_step))
    # else:
    #   print('Exception and Exit loading process')
    #   return
    # prediction = session.run(pred)
    # sess = tf.InteractiveSession()
    # saver = tf.train.Saver()
    # saver.restore(sess, tf.train.latest_checkpoint('model'))
    # print('Complete load model')
    

    # predictedSentiment = sess.run(prediction, {input_data:test_data})
    # print("Checking predictedSentiment ... ", predictedSentiment)
    # if (predictedSentiment[0])>(predictedSentiment[1]):
    #   print("Positive value is ", predictedSentiment[0])
    #   return 1
    # else:
    #   print("Negative value is ", predictedSentiment[1])
    #   return 0


  @classmethod
  def __getSentenceMatrix(self, string):
    cleanedSentence = self.__cleanSentences(string)
    split = cleanedSentence.split()
    batch = []
    matrix = []
    for word in split:
      if word:
        word_vector = em.where('word', word).first().vector
        matrix.append(word_vector)
      else:
        matrix.append([0] * 300)

    for l in range(len(matrix), ts.maxSentenceLen() - 1):
      matrix.append([0] * 300)
    batch.append(matrix)
    tf.stack(np.asarray(batch))
    #print('Done Sentence Matrix ===')
    #print(batch[0][1])
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