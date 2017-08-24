from ..models import TrainingSet, EmMatrix
from .base import Base
from flask_restful import Resource
from ..models import TrainingSet
from ..config import batchSize, redis
from .. import db_conn
from ..models import TrainingSet as ts
from ..models import EmMatrix as em

import tensorflow as tf


class Trainer(Base):

    @classmethod
    def cleanSentences(cls, string):
        strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
        string = string.lower().replace("<br />", " ")
        string = re.sub(strip_special_chars, "", string.lower())
        return string

    @classmethod
    def createGraph(cls,data):

        tf.reset_default_graph()

        labels = tf.placeholder(tf.float32, [batchSize, TrainingSet.maxSentenceLen()], name = 'labels_placeholder')
        input_data = tf.placeholder(tf.int32, [batchSize, TrainingSet.maxSentenceLen()], name = 'input_placeholder')

        # data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]), dtype = tf.float32)
        # data = tf.nn.embedding_lookup(em,input_data)

        lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
        lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
        value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

        weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
        bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
        value = tf.transpose(value, [1, 0, 2])
        last = tf.gather(value, int(value.get_shape()[0]) - 1)
        prediction = (tf.matmul(last, weight) + bias)

        correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
        accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
        optimizer = tf.train.AdamOptimizer().minimize(loss)
        
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
    def getModel(cls, iterations,data,data_labels):
        sess = tf.InteractiveSession
        prediction, optimizer = cls.createGraph(batchSize, data)
        for i in range(iterations):
            #nextBatch, nextBatchLabels = cls.getTrainBatch(batchSize, TrainingSet.maxSentenceLen())
            sess.run(optimizer, {input_data:data, labels:data_labels})

            if i % 90000 == 0 and i != 0:
                save_path = saver.save(sess, "prediction/trained_lstm.ckpt", global_step = i)
                print("Trained Done!")
        writer.close()
            



