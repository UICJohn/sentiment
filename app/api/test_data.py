import tensorflow as tf 
import numpy as np 
import re


class test_step():
    """docstring for test"""


    def cleanSentence(string):
        strip_special_chars = re.compile("[^A-Za-z0-9]+")
        string = string.lower().replace("<br />", " ")
        string = re.sub(strip_special_chars, "", string.lower())
        return string

    def test_data(string):
        #import model
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        saver.restore(sess, sess, tf.train.latest_checkpoint('models/pretrained_lstm.ckpt'))


        test_arr = np.zeros([batchSize, maxSeqLength])
        sentenceMat = np.zeros([batchSize, maxSeqLength], dtype ='int32')
        cleanedSentence = cleanSentence(string)
        split = cleanedSentence.split()

        for indexCounter,word in enumerate(split):
            pass

        
    
        