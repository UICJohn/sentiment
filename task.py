from app import init_celery, sentiment_app
from app.lib import Batch
from app.lib import Trainer
celery = init_celery(sentiment_app)

import numpy as np
import tensorflow as tf

@celery.task()
def create_batch(quantity = 1):
	Batch.enqueue(quantity)

@celery.task()
def train():
	#pass
	#从redis 拿数据，放到trainer
	#batch 格式 [batch, labels]
	batches = Batch.dequeue()
	#Trainer.getModel(tf.stack(np.asarray(batches[0])),tf.stack(np.asarray(batches[1])))
	#Trainer.getModel(np.asarray(batches[0]),np.asarray(batches[1]))
	Trainer.getModel(tf.convert_to_tensor(np.asarray(batches[0]),dtype=np.float32),tf.convert_to_tensor(np.asarray(batches[1]),dtype=np.float32))
	# print("#######", np.asarray(batches[0]).shape)
	# print("#######", np.asarray(batches[1]).shape)
	# print("++++++++++", tf.stack(np.asarray(batches[0])))
	# print("++++++++++", tf.stack(np.asarray(batches[1])))
		#print(batches)

