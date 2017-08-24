from ..models import TrainingSet, EmMatrix
from .base import Base
from flask_restful import Resource
from ..models import TrainingSet
from ..config import batchSize, redis
from .. import db_conn
from ..models import TrainingSet as ts
from ..models import EmMatrix as em

import tensorflow as tf
import numpy as np

tf.reset_default_graph()
lstmUnits = 64
numClasses = 3

class Trainer(Base):

    @classmethod
    def cleanSentences(cls, string):
        strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
        string = string.lower().replace("<br />", " ")
        string = re.sub(strip_special_chars, "", string.lower())
        return string

    @classmethod
    def createGraph(cls,batchSize, data,data_labels,iterations=100000):

        #tf.reset_default_graph()
        
        labels = tf.placeholder(tf.float32, [batchSize, numClasses], name = 'labels_placeholder')
        input_data = tf.placeholder(tf.int32, [batchSize, TrainingSet.maxSentenceLen(), 300], name = 'input_placeholder')
        print(" Coming into create graph   line 31", data_labels)

        # data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]), dtype = tf.float32)
        # data = tf.nn.embedding_lookup(em,input_data)
        #print("The input data shape  is #######################" , labels)
        #print("The data shape is ####################### ", data)
        lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
        #print(" Coming into create graph   line 36")
        lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
        #print(" Coming into create graph   line 38")
        value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
        #print(" Coming into create graph   line 40")
        weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
        #print(" Coming into create graph   line 42")
        bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
        #print(" Coming into create graph   line 44")
        value = tf.transpose(value, [1, 0, 2])
        #print(" Coming into create graph   line 46")
        last = tf.gather(value, int(value.get_shape()[0]) - 1)
        #print(" Coming into create graph   line 48")
        prediction = (tf.matmul(last, weight) + bias)
        #print(" Coming into create graph   line 50")
        correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
        #print(" Coming into create graph   line 52")
        accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
        #print(" Coming into create graph   line 54")
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
        #print(" Coming into create graph   line 56")
        optimizer = tf.train.AdamOptimizer().minimize(loss)
        #print(" Coming into create graph   line 58")

        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        print("the data type is +++++++++++++++++++  ", data.eval().shape)
        for i in range(iterations):
            print("Start to train model  ", i)
            #print("satrt to train model in line 67",data.eval())
            sess.run(optimizer, {input_data:data.eval(),labels:data_labels.eval()})
            if i % 90000 == 0 and i != 0:
                save_path = saver.save(sess, "prediction/trained_lstm.ckpt", global_step = i)
                print("Trained Done!")
        writer.close()
        return prediction,optimizer

    # @classmethod
    # def getTrainBatch(cls, batchSize, maxSeqLength):
    #     labels = []
    #     train_set = np.zeros([batchSize, maxSeqLength])
    #     for i in range(batchSize):
    #         if (i%2 == 0):
    #             num = randint(1,12472)
    #             labels.append([1,0])
    #         else:
    #             num = randint(12473,24903)
    #             labels.append([0,-1]) 
    #         train_set[i] = ts.where('id', num).first().word_ids

    #     return train_set, labels


    @classmethod
    def getModel(cls, data,data_labels,iterations=100000):
        #sess = tf.InteractiveSession
        prediction, optimizer = cls.createGraph(batchSize, data,data_labels,iterations=100000)
        # for i in range(iterations):
        #     print("Start to train model  ", i)
        #     #nextBatch, nextBatchLabels = cls.getTrainBatch(batchSize, TrainingSet.maxSentenceLen())
        #     sess.run(optimizer, {data:data, labels:data_labels})

        #     if i % 90000 == 0 and i != 0:
        #         save_path = saver.save(sess, "prediction/trained_lstm.ckpt", global_step = i)
        #         print("Trained Done!")
        # writer.close()
            



