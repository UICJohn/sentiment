from ..models import TrainingSet, EmMatrix
from .base import Base
from flask_restful import Resource
from ..models import TrainingSet
from ..config import batchSize, redis
from .. import db_conn
from ..models import TrainingSet as ts
from ..models import EmMatrix as em
from ..config import redis
from ..config import batchSize, numClasses, lstmUnits, cluster_spec, max_epoch
from .batch import Batch

import numpy as np

import tensorflow as tf


class Classifier(Base):

    @classmethod
    def getSentenceMatrix(cls, string):
        cleanedSentence = cls.cleanSentences(string)
        split = cleanedSentence.split()
        batch = []
        sentenceMat = np.zeros([batchSize, ts.maxSentenceLen()], dtype = 'int32'])
        matrix = []
        for word in enumerate(split):
            if word:
                word_vector = em.where('word', word).first().vector
                matrix.append(word_vector)
            else:
                matrix.append([0] *300)
        for l in range(len(matrix), ts.maxSentenceLen() - 1):
            matrix.append([0] * 300)
        batch.append(matrix)
        tf.stack(np.asarray(batch))
        print("the input batch shape is ", batch.shape)
        return batch

    @classmethod
    def createOP(cls):
        weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
        bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
      
        print("create input")
        #Input
        input_data = tf.placeholder(tf.float32, [batchSize, TrainingSet.maxSentenceLen(), 300], name = 'input_placeholder')
        labels = tf.placeholder(tf.float32, [batchSize, numClasses], name = 'labels_placeholder')

        print("global step")
        #global_step
        global_step = tf.contrib.framework.get_or_create_global_step()

        print("initial lstm cell")
        #initial lstm cell
        lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
        lstmCell = tf.contrib.rnn.DropoutWrapper(cell = lstmCell, output_keep_prob=0.75)

        print("Done graph")
        #finalize graph
        outputs, _ = tf.nn.dynamic_rnn(lstmCell, input_data, dtype=tf.float32)

        print("loss and accuracy")
        #define loss and accuracy
        outputs = tf.transpose(outputs, [1, 0, 2])
        last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)
        prediction = (tf.matmul(last, weight) + bias)
        correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
        accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
        optimizer = tf.train.AdamOptimizer().minimize(loss, global_step = global_step)

        return prediction,optimizer



    @classmethod
    def cleanSentences(cls,string):
        strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
        string = string.lower().replace("<br />", " ")
        string = re.sub(strip_special_chars, "", string.lower())
        return string

    @classmethod
    def classification(cls, string):
        prediction = cls.createOP()

        sess = tf.InteractiveSession()
        saver = tf.train.Saver()

        saver.restore(sess, tf.train.latest_checkpoint('prediction'))

        test_data = cls.getSentenceMatrix(string)
        tf.stack(np.asarray(test_data))

        predictedSentiment = sess.run(prediction, {input_data: test_data})
        print("#####################", predictedSentiment)
        if (predictedSentiment[0])>(predictedSentiment[1]):
            print ("Positive Sentiment")
            return 1
        else:
            print ("Negative Sentiment")
            return 0

        

