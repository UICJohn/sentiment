from flask_restful import Resource
from ..models import TrainingSet
from ..config import batchSize, redis
from .. import db_conn
from ..models import TrainingSet as ts
from ..models import EmMatrix as em

import tensorflow as tf


class TrainingSetController(Resource):

  @classmethod
  def cleanSentences(string):
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    string = re.sub(strip_special_chars, "", string.lower())
    return string
  
  @classmethod
  def createGraph(batchSize, numClasses, maxSeqLength, numDimensions, em):
    tf.reset_default_graph()

    labels = tf.placeholder(tf.float32, [batchSize, numClasses], name = 'labels_placeholder')
    input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength], name = 'input_placeholder')

    data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]), dtype = tf.float32)
    data = tf.nn.embedding_lookup(em, input_data)

    print("The em shape is ", em.shape)


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

    # for i in range(100000):
    #   nextBatch, nextBatchLabels = getTrainBatch(batchSize,maxSeqLength)
    #   #sess.run(optimizer, {input_data: nextBatch, labels:nextBatchLabels})



  @classmethod
  def getTrainBatch(batchSize, maxSeqLength):
    labels = []
    train_set = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
      if (i%2 == 0):
        num = randint(1,12472)
        labels.append([1,0])
      else:
        num = randint(12473,24903)
        labels.append([0,1]) 
      train_set[i] = ts.where('id', num).first().word_ids

    return train_set, labels


  def get(self):
    if redis.has('max_word_length'):
      redis.get('max_word_length')
    else:
      TrainingSet.select(db_conn.raw("max(array_length(regexp_split_to_array(words, '\s|\t'), 1)) as max_len")).first().max_len
      redis.put('max_word_length', max_word_length, 180)
    set_info = {'Count': TrainingSet.count(), 'Maximum Word Length': redis.get('max_word_length'), 'Batch Size': batchSize, 'Batch Count': 2}
    return set_info



  def post(self):
  	# TODO add trainning set


  def getModel(self):







