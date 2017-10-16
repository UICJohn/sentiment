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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

      prediction = tf.matmul(last, weight) + bias
      correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
      accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
      train_step = (tf.train.AdamOptimizer().minimize(loss, global_step = global_step))
      saver = tf.train.Saver()
    



    # test_data, labels_temp, numWords = self.__getTestMatrix()
    # numFiles = len(numWords)
    
    #test_data_neg, test_data_pos = self.__getTestMatrix()
    #test_data = self.__getTestMatrix('/Users/chih/Documents/IOS/dev_sentiment/sentiment/neg/')
    test_data = self.eachFile('/Users/chih/Documents/IOS/dev_sentiment/sentiment/pos/')

    print("line 61 -------------  ", len(test_data))
    #print("line 62 -------------  ", test_data[1])
    #print('\n')
    #print("line 62 -------------  ", len(test_data[1]))
    #print('line 62 ++++++++++++++++', len(test_data[1]))
    #print('line 63 ================', test_data[0][0])
    correct_result  = []
    incorrect_result = []
    prediction_result = []
    final_result = ''
    with tf.Session(graph = graph) as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state('logs')
        probability_value = 0
        if ckpt and tf.gfile.Exists('logs'):
            print('Start load model')
            saver.restore(sess, ckpt.model_checkpoint_path)
            for i in range(0,len(test_data)):
              temp_input = test_data[i]
              predictedSentiment = sess.run(tf.nn.softmax(prediction), {input_data:test_data[i]})[0]
              if (predictedSentiment[0])>(predictedSentiment[2]):
                print('The line 83 --------------------', predictedSentiment)

                prediction_result.append(0)
              else:
                print('The line 87 --------------------', predictedSentiment)
                prediction_result.append(1)
        else:
            print('no checkpoint found') 
            return
    print("total files is  -------", prediction_result)
    #print('Negative label is ', prediction_result.count(1))
    #self.__drawGraph([len(correct_result)/ 800, len(correct_result)/800], 'neg_result_.png')


    # Describe sequence length
    # print('The Pos numFiles is ', neg_files)
    # print('The Neg numFiles is ', pos_files)

    # plt.hist(neg_files,50)
    # plt.xlabel('Sequence Length')
    # plt.ylabel('Frequency')
    # plt.axis([0, 1000,0, 1200])
    # plt.plot()
    # plt.savefig('test_neg_files.png')
    # plt.close()

    # plt.hist(pos_files,50)
    # plt.xlabel('Sequence Length')
    # plt.ylabel('Frequency')
    # plt.axis([0, 1000,0, 1200])
    # plt.plot()
    # plt.savefig('test_pos_files.png')
    # self.__drawGraph([80,30,40,50])

    #print("================================= start to save files")
    #np.save("test_matrix_neg.npy",test_data_neg[0])
    #print("================================= Done to save neg")
    #np.save("test_matrix_label_neg.npy", test_data_neg[1])
    #np.save("test_matrix_pos.npy",test_data_pos[0])
    #np.save("test_matrix_label_pos.npy", test_data_pos[1])
    #print('Done to save test data')
    # with tf.Session(graph = graph) as sess:
    #   sess.run(tf.global_variables_initializer())
    #   ckpt = tf.train.get_checkpoint_state('logs')
    #   if ckpt and tf.gfile.Exists('logs'):
    #     saver.restore(sess, ckpt.model_checkpoint_path)
    #     a = np.load("test_matrix.npy")
    #     #print('~~~~~~~~~~~~~~~~~', labels)
    #     for i in range(1,25):
    #       print("Accuracy for this batch:", (sess.run(accuracy, {input_data: a[0], labels : labels_temp})) * 100 )
    #     # sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels}) * 100
    #   else:
    #     print('No checkpoint file')
    # np.save("test_matrix.npy",test_data)



  # @classmethod
  # def __getTestMatrix(self, data_path):
  #   print('Preparing test data')
  #   test_data,numWords = self.eachFile(data_path)
  #   return test_data

  @classmethod
  def eachFile(self, filePath):
    num = 0
    test_data = []
    #batch = []
    files = [filePath + f for f in listdir(filePath) if isfile(join(filePath, f))]
    labels = []
    numWords = []
    for f in files:
      num = num + 1
      if num < 1001:
        print(f)
        with open(f, "r", encoding = 'utf-8') as f:
          #read each file
          line = f.readline()
          line = self.__cleanSentences(line)
          #print('line 162-----------------', line)
          split = line.split()
          #print('line 163______________', split)
          matrix = []
          counter = len(split)
          numWords.append(counter)
          
          for word in split:#convert one text to word matrix // each matrix contains word vector of each wrod in one text file 
            #print("Coming in if word :")
            #print(word)
            if word:
              #print('Coming in if word:')
              if em.where('word', word).first():
                #print('Could find word in embedding matrix')
                word_vector = em.where('word', word).first().vector
                #print("The found word vector is ", word_vector)
                matrix.append(word_vector)
              else:
                #print('Could not find word in embedding matrix')
                #word_vector.append([0] * 300)
                matrix.append([0] * 300)
                #print("Couldn't found word vector is ", word_vector)
          #matrix.append(word_vector)

          if len(split) >= TrainingSet.maxSentenceLen():
            #print('The length of input text >>>>>>>>>>>>>>>>>>>>>>>> Max Sentence length')
            matrix = matrix[0: TrainingSet.maxSentenceLen()]
            batch =[]
            for i in range(0, batchSize):
              batch.append(matrix)
            test_data.append(batch)
          else:
            #print('The length of input text <<<<<<<<<<<<<<<<<<<<<<<< Max Sentence length')
            for l in range(len(matrix), TrainingSet.maxSentenceLen()):
              matrix.append([0] * 300)
              batch =[]
            for i in range(0,batchSize):
              #print('-------------------------------------    199 i is ', i)
              batch.append(matrix)
            test_data.append(batch)
            #print("Line 201 batch size is ", len(batch))
            #print("Line 193 batch size is ", len(batch))
    return test_data
   
  # @classmethod
  # def eachFile(self,filePath):
  #   num = 0
  #   test_data = []
  #   batch = []
  #   files = [filePath + f for f in listdir(filePath) if isfile(join(filePath, f))]
  #   labels = []
  #   numWords = []
  #   for f in files:
  #     num = num + 1
  #     if (num < 2):
  #       with open(f, "r", encoding = 'utf-8') as f:
  #         line = f.readline()
  #         line = self.__cleanSentences(line)
  #         split = line.split()
  #         counter = len(split)
  #         numWords.append(counter)
  #         print("____________", split)
  #         for word in split:
  #           matrix = []
  #           if word:
  #             if em.where('word',word).first():
  #               word_vector = em.where('word', word).first().vector
  #               matrix.append(word_vector)
  #               print("each word is ", word_vector)
  #             else:
  #               word_vector.append([0] * 300)
  #             matrix.append(word_vector)
  #           else:
  #             matrix.append([0] * 300)
  #         print('++++++++++++++++++++++++++', len(matrix))
  #         for l in range(len(matrix), TrainingSet.maxSentenceLen()):
  #           matrix.append([0] * 300)
  #         for i in range(1,batchSize+1):
  #           batch.append(matrix)
  #           labels.append([0,0,1])
  #         test_data.append(batch)
  #       print('The num is', num)
  #   return test_data, labels, numWords


  @classmethod
  def __cleanSentences(self, string):
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    string = re.sub(strip_special_chars, "", string.lower())
    return string


  @classmethod
  def __drawGraph(self, prediction_result, name):
    plt.figure(figsize = (6,9))
    labels = [u'Pos_correct', u'Neg_correct', u'Pos_incorrect', u'Neg_incorrect']
    colors = ['red', 'yellowgreen', 'lightskyblue', 'pink']
    explode = (0.05,0.05,0,0)
    patches, l_text, p_text = plt.pie(prediction_result, explode = explode, labels = labels, colors = colors, labeldistance = 1.1, autopct = '%3.1f%%', shadow = False,
                                startangle = 90,pctdistance = 0.6)

    plt.axis('equal')
    plt.legend()
    plt.plot()
    plt.savefig(name)





