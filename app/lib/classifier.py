from ..models import TrainingSet, EmMatrix
from .base import Base
from flask_restful import Resource
from ..models import TrainingSet
from ..config import batchSize, redis
from .. import db_conn
from ..models import TrainingSet as ts
from ..models import EmMatrix as em

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
    def cleanSentences(cls,string):
        strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
        string = string.lower().replace("<br />", " ")
        string = re.sub(strip_special_chars, "", string.lower())
        return string


    @classmethod
    def classification(cls, string):
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('prediction'))

        test_data = cls.getSentenceMatrix(string)
        tf.stack(np.asarray(test_data))

        predictedSentiment = sess.run(prediction, {input_data: test_data})
        if (predictedSentiment[0])>(predictedSentiment[1]):
            print ("Positive Sentiment")
            return 1
        else:
            print ("Negative Sentiment")
            return 0

        

