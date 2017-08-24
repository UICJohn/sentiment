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
	batches = Batch.dequeue()
	Trainer.getModel(tf.convert_to_tensor(np.asarray(batches[0]),dtype=np.float32),tf.convert_to_tensor(np.asarray(batches[1]),dtype=np.float32))