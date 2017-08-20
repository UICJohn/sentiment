from os import listdir
from os.path import isfile, join
from random import randint

import tensorflow as tf 
import numpy as np 
import re 
import types 
import datetime


class train_step():
    """docstring for ClassName"""
    def cleanSentence(string):
        strip_special_chars = re.compile("[^A-Za-z0-9]+")
        string = string.lower().replace("<br />", " ")
        string = re.sub(strip_special_chars, "", string.lower())
        return string


    def generateIdsMatrix(postive_path,negative_path):
        idsMatrix = np.zeros((numFiles, maxSeqLength), dtype = 'int32')
        fileCounter = 0

        positiveFiles = [postive_path + f for f in listdir(postive_path) if isfile(join(postive_path,f))]

        negativeFiles = [negative_path + f for f in listdir(negative_path) if isfile(join(negative_path,f))]

        for pf in positiveFiles:
            with open(pf, "r") as f:
                indexCounter = 0
                line = f.readline()
                cleanedLine = cleanSentence(line)
                split = cleanedLine.split()
                for word in split:
                    try:
                        idsMatrix[fileCounter][indexCounter] = wordsList.index(word)
                    except ValueError:
                        idsMatrix[fileCounter][indexCounter] = wordVectors.shape[0] - 1
                    indexCounter = indexCounter + 1
                    if indexCounter >= maxSeqLength:
                        break
                fileCounter = fileCounter + 1

        for nf in negativeFiles:
            with open(nf, "r") as f:
                indexCounter = 0
                line=f.readline()
                cleanedLine = cleanSentence(line)
                split = cleanedLine.split()
                for word in split:
                    try:
                        idsMatrix[fileCounter][indexCounter] = wordsList.index(word)
                    except ValueError:
                        idsMatrix[fileCounter][indexCounter] = wordVectors.shape[0] - 1
                    indexCounter = indexCounter+ 1
                    if indexCounter >= maxSeqLength:
                        break
                fileCounter = fileCounter + 1

        np.save('idsMatrix', ids)



    def getTrainBatch():
        labels = []
        train_data = np.zeros([batchSize, maxSeqLength])

        for i in range(batchSize):
            if (i%2 == 0):
                num = randint(1,11499)
                labels.append([1,0])
            else:
                num = randint(12500, 24999)
                labels.append([1,0])
            train_data[i] = ids[num-1:num]

        return train_data, labels




    def createGraph():
        tf.reset_default_graph()
        labels = tf.placeholder(tf.float32, [batchSize, numClasses])
        input_data = tf.placeholder(tf.float32, [batchSize, maxSeqLength])

        data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
        data = tf.nn.embedding_lookup(wordVectors,input_data)

        lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
        lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.25)
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

    
    def trainingModel(iterations, optimizer):
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        for i in range(iterations):
            nextBatch, nextBatchLabels = getTrainBatch()
            sess.run(optimizer, {input_data:nextBatch, labels: nextBatchLabels})

            if (i%50 == 0):
                summary = sess.run(merged, {input_data:nextBatch, labels:nextBatchLabels})
                writer.add_summary(summary,i)

            if (i% 10000 == 0 and i != 0):
                save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
                print("Completed train and saved model in %s", save_path)
        writer.close()
            










