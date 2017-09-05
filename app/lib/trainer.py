# Between Graph Training
from .base import Base
from ..models import TrainingSet
from ..config import batchSize, numClasses, lstmUnits, cluster_spec
from .batch import Batch
import tensorflow as tf
import numpy as np
from .task_control import TaskControl
import os

class Trainer(Base):
  def __init__(self, task_type= 'worker'):
    tc = TaskControl(task_type = task_type)
    self.task_index = tc.add_task()
    print(self.task_index)
    self.task_type = task_type
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(self.task_index)

  def process(self):
    print("Processing...")
    cluster = tf.train.ClusterSpec(cluster_spec)
    print("task_type: " + str(self.task_type) + " task_index: " + str(self.task_index))
    server = tf.train.Server(cluster, job_name = self.task_type, task_index = self.task_index)
    if(self.task_type == 'ps'):
      print("PS TASK")
      server.join()
    else:
      print("Start Training")
      Batch.enqueue()
      while True:
        batch = Batch.dequeue(timeout = 1000*60*4)
        if not batch:
          print("Training Done")
          break
        else:
          print("Training Begin")
          self.__create_graph(batchSize,tf.convert_to_tensor(np.asarray(batch[0]),dtype=np.float32),tf.convert_to_tensor(np.asarray(batch[1]),dtype=np.float32), server= server, cluster= cluster, iterations=100000)
          Batch.enqueue()

  def __create_graph(self, batchSize, data, data_labels, server, cluster, iterations=100000):
    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:"+str(self.task_index), cluster=cluster)):
      print("Creating graph")
      weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
      bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
      input_data = tf.placeholder(tf.int32, [batchSize, TrainingSet.maxSentenceLen(), 300], name = 'input_placeholder')
      lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
      lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
      value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
      value = tf.transpose(value, [1, 0, 2])
      labels = tf.placeholder(tf.float32, [batchSize, numClasses], name = 'labels_placeholder')
      last = tf.gather(value, int(value.get_shape()[0]) - 1)
      prediction = (tf.matmul(last, weight) + bias)
      correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
      accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
      global_step = tf.contrib.framework.get_or_create_global_step()
      optimizer = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)
      print("Graph created")
    hooks=[tf.train.StopAtStepHook(last_step=iterations)]
    with tf.train.MonitoredTrainingSession(master = server.target, is_chief=(self.task_index == 0), checkpoint_dir="/tmp", hooks=hooks) as sess:
      counter = 0
      while not sess.should_stop():
        sess.run(optimizer, {input_data:data.eval(session= sess),labels:data_labels.eval(session=sess)})
        if (counter%1000 == 0):
          print("Task %d Iteration Times: %d" %(self.task_index, counter))
        counter += 1