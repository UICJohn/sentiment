# Between Graph Training
from .base import Base
from ..models import TrainingSet
from ..config import redis
from ..config import batchSize, numClasses, lstmUnits, cluster_spec, max_epoch
from .batch import Batch
import tensorflow as tf
import datetime
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
      self.__process_graph(server= server, cluster= cluster)
      self.__process_graph(server = task, cluster = cluster)
  def __process_graph(self, server, cluster):
    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:"+str(self.task_index), cluster=cluster)):
      #create graph
      weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
      bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
      #Input
      input_data = tf.placeholder(tf.float32, [batchSize, TrainingSet.maxSentenceLen(), 300], name = 'input_placeholder')
      labels = tf.placeholder(tf.float32, [batchSize, numClasses], name = 'labels_placeholder')
      #global_step
      global_step = tf.contrib.framework.get_or_create_global_step()
      #initial lstm cell
      lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
      lstmCell = tf.contrib.rnn.DropoutWrapper(cell = lstmCell, output_keep_prob=0.75)
      #finalize graph
      outputs, _ = tf.nn.dynamic_rnn(lstmCell, input_data, dtype=tf.float32)
      #define loss and accuracy
      outputs = tf.transpose(outputs, [1, 0, 2])
      last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)
      prediction = (tf.matmul(last, weight) + bias)
      tf.add_to_collection('pred_network', prediction)
      correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
      accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
      op = tf.train.AdamOptimizer().minimize(loss, global_step = global_step)

      



      print("Tensorboard parameters")

      tf.summary.scalar('Loss', loss)
      tf.summary.scalar('Accuracy', accuracy)
      tf.summary.histogram('weight',weight)
      tf.summary.histogram('bias', bias)
      summary_op = tf.summary.merge_all()
      logdir = "tensorBoard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
      # summary_hook = tf.train.SummarySaverHook(save_secs=600,output_dir=logdir,summary_op=summary_op)
      
      print("Done prepare tensorboard parameters")

      for i in range(0, max_epoch):
        if(self.task_index == 0):
          print("CURRENT EPOCH: %d" % i)
        hooks=[tf.train.StopAtStepHook(last_step = 1000 * (i + 1))]
        #summary_hook = tf.train.SummarySaverHook(save_secs=600,output_dir=logdir,summary_op=summary_op) 
        #with tf.train.MonitoredTrainingSession(master = server.target, is_chief=(self.task_index == 0), checkpoint_dir= os.path.expanduser('~/sentiment/logs/'), hooks = hooks, chief_only_hooks = summary_hook) as sess:

        with tf.train.MonitoredTrainingSession(master = server.target, is_chief=(self.task_index == 0), checkpoint_dir= os.path.expanduser('~/sentiment/logs/'), hooks = hooks) as sess:
          step_count = 0
          print("Start to summary graph")
          
          #writer = tf.summary.FileWriter(logdir, sess.graph)

          while not sess.should_stop():
            print("In while not sess.should_stop()")
            training_set_ids = Batch.dequeue()
            data, data_labels = self.vector2matrix(training_set_ids)
            sess.run(op, {input_data: data, labels: data_labels})
            
            step_count += 1
            
            print("Start to summary ops")

            # if (i%100 == 0 and i > 0):
            #   summary = sess.run(summary_op, {input_data: data, labels:data_labels})
            #   print("Start to add summary files")
            #   writer.add_summary(summary, i)
            # writer.close()
                          
          print("%d Training Done" % self.task_index)
          







