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
import random
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
      lstmCell = tf.contrib.rnn.DropoutWrapper(cell = lstmCell, output_keep_prob=1)
      outputs, _ = tf.nn.dynamic_rnn(lstmCell, input_data, dtype=tf.float32)

      outputs = tf.transpose(outputs, [1, 0, 2])
      last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

      prediction = tf.matmul(last, weight) + bias
      correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
      accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
      train_step = (tf.train.AdamOptimizer().minimize(loss, global_step = global_step))
      saver = tf.train.Saver()

    #test_data = self.eachFile('/Users/chih/Documents/IOS/dev_sentiment/sentiment/Neg_kayla/')

    #print("line 61 -------------  ", len(test_data))

    correct_result  = []
    incorrect_result = []
    prediction_result = []
    final_result = ''
    # with tf.Session(graph = graph) as sess:
    #     sess.run(tf.global_variables_initializer())
    #     ckpt = tf.train.get_checkpoint_state('logs')
    #     probability_value = 0
    #     if ckpt and tf.gfile.Exists('logs'):
    #         print('Start load model')
    #         saver.restore(sess, ckpt.model_checkpoint_path)
    #         for i in range(0,len(test_data)):
    #           temp_input = test_data[i]
    #           predictedSentiment = sess.run(tf.nn.softmax(prediction), {input_data:test_data[i]})[0]
    #           if (predictedSentiment[0])>(predictedSentiment[2]):
    #             print('The line 83 --------------------', predictedSentiment)

    #             prediction_result.append(0)
    #           else:
    #             print('The line 87 --------------------', predictedSentiment)
    #             prediction_result.append(1)
    #     else:
    #         print('no checkpoint found') 
    #         return
    # output_neg = prediction_result.count(0)
    # output_pos = prediction_result.count(1)
    # self.__drawGraph([output_pos,output_neg], 'pos_kayla_result_.png')
     
    # output_neg = prediction_result.count(0)
    # output_pos = prediction_result.count(1)
    # self.__drawGraph([output_pos,output_neg], 'pos_kayla_result_.png')
    print('Start to test in line 84')
    
    ls1 = self.__randomTest('/Users/chih/Documents/IOS/sentiment_dev/sentiment/test_pos/','/Users/chih/Documents/IOS/sentiment_dev/sentiment/test_neg/', 'logs_0611')
    ls2 = self.__randomTest('/Users/chih/Documents/IOS/sentiment_dev/sentiment/Positive_kayla/','/Users/chih/Documents/IOS/sentiment_dev/sentiment/Neg_kayla/', 'logs_0611')
    # [str(len(reorder_label)), str(correct_result), str(len(reorder_label) - correct_result)]
    self.__drawGraph(ls1,ls2,'data_correct_result_analysis.png')

  @classmethod
  def eachFile(self, filePath):
    num = 0
    test_data = []
    files = [filePath + f for f in listdir(filePath) if isfile(join(filePath, f)) and not f.startswith('.')]
    labels = []
    # print("----------------------------", len(files))
    numWords = []
    for f in files:
      num = num + 1
      if num < len(files)+1:
        # print(f)
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
    return test_data

  @classmethod
  def __cleanSentences(self, string):
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    string = re.sub(strip_special_chars, "", string.lower())
    return string

  @classmethod
  def __drawGraph(self, ls1,ls2, name):
    # total_data, correct_data, incorrect_data
    print("================================", ls1, " ----- ", ls2)
    plt.figure(figsize = (6,9))
    ax1 = plt.subplot(211)

    labels = [u'Correct: ' + str(ls1[1]), u'Incorrect: '+ str(ls1[2])]
    colors = ['red', 'yellowgreen']
    explode = (0.05,0.05)
    patches, l_text, p_text = plt.pie([ls1[1], ls1[2]], explode = explode, labels = labels, colors = colors, labeldistance = 1.1, autopct = '%3.1f%%', shadow = False,
                              startangle = 90,pctdistance = 0.6) 
    # plt.pie(prediction_result, labels = labels, autopct = '%1.2f%%')
    plt.axis('equal')
    plt.title("Truth Data: " + str(ls1[0]))
    plt.legend(loc='upper left',fancybox=True,shadow=True)

    ax2 = plt.subplot(212)
    labels = [u'Correct: ' + str(ls2[1]), u'Incorrect: '+ str(ls2[2])]
    colors = ['red', 'yellowgreen']
    explode = (0.05,0.05)
    patches, l_text, p_text = plt.pie([ls2[1], ls2[2]], explode = explode, labels = labels, colors = colors, labeldistance = 1.1, autopct = '%3.1f%%', shadow = False,
                              startangle = 90,pctdistance = 0.6) 
    # plt.pie(prediction_result, labels = labels, autopct = '%1.2f%%')
    plt.axis('equal')
    plt.title("Truth Data: " + str(ls2[0]))
    plt.legend(loc='upper left',fancybox=True,shadow=True)

    plt.plot()
    plt.savefig(name)

  @classmethod
  def __randomTest(self, pos_folder, neg_folder, checkpoint_file):
    
    # print('Stop')

    pos_data = self.eachFile(pos_folder)
    neg_data = self.eachFile(neg_folder)
    print('-=====================================', len(pos_data))
    print('-=====================================', len(neg_data))
    # print("neg_data and pos_data includes files : ", len(pos_data), " .......... ", len(neg_data))
    neg_label = [0] * len(neg_data)
    pos_label = [1] * len(pos_data)
    dataSet = pos_data+ neg_data
    labelSet = pos_label + neg_label

    s = list(range(len(dataSet)))
    print("================================== before random", s)
    random.shuffle(s)
    print("================================== after random", s)

    reorder_data = []
    reorder_label = []
    prediction_result = []

    for i in range(0, len(s)):
      reorder_data.append(dataSet[s[i]])
      reorder_label.append(labelSet[s[i]])
    
    # print("before reorder, the actual data label is ---", labelSet)
    # print("after reorder, the actual data label is +++", reorder_label)

    # print("after reorder, the neg data and pos data include files : ", len(pos_data), ".........", len(neg_data))
    # print("after reorder, the neg label and pos label include files : ", len(pos_label), ".........", len(neg_label))
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

    with tf.Session(graph = graph) as sess:
      sess.run(tf.global_variables_initializer())
      ckpt = tf.train.get_checkpoint_state(checkpoint_file)

      if ckpt and tf.gfile.Exists(checkpoint_file):
        print("Start load model")
        saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(0,len(reorder_data)):
          # temp_input = reorder_data[i]
          predictedSentiment = sess.run(tf.nn.softmax(prediction), {input_data: reorder_data[i]})[0]
          if (predictedSentiment[0])>(predictedSentiment[2]):
            # print('The line 203 --------------------', predictedSentiment)
            prediction_result.append(0)
          else:
            # print('The line 205 --------------------', predictedSentiment)
            prediction_result.append(1)
      else:
        print('no checkpoint found')
        return
    correct_result = 0
    
    incorrect_data_index = []
    for i in range(0,len(reorder_label)):
      if reorder_label[i] == prediction_result[i]:
        correct_result = correct_result + 1
      else:
        incorrect_data_index.append(s[i])

    #print("The correct result is #################### ", correct_result)
    #print("Actual the label is ++++++++++++++++++++++ ", reorder_label, "++++++++++++++ ", len(reorder_label) )
    #print("The prediction result is ----------------- ", prediction_result, "-----------", len(prediction_result))

    # print("\n")
    print("Incorrect data index is ############ ", incorrect_data_index)
    print("\n")
    self.__searchFiles(pos_folder,neg_folder, incorrect_data_index)
    return [len(reorder_label), correct_result, len(reorder_label) - correct_result]
    # return [str(len(reorder_label)), str(correct_result), str(len(reorder_label) - correct_result)]

    # self.__drawGraph([correct_result, len(reorder_label) - correct_result],'data_correct_result_analysis.png',str(len(reorder_label)), str(correct_result), str(len(reorder_label) - correct_result))
          
  @classmethod
  def __searchFiles(self, folder1_path, folder2_path, incorrect_data_index):
    #print
    files1 = [folder1_path + f for f in listdir(folder1_path) if isfile(join(folder1_path, f)) and not f.startswith('.')]
    files2 = [folder2_path + f for f in listdir(folder2_path) if isfile(join(folder2_path, f)) and not f.startswith('.')]
    files = files1 + files2
    for f in range(0, len(incorrect_data_index)):
      print("Incorrect data index is ############", files[incorrect_data_index[f]])




