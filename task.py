from app import init_celery, sentiment_app
from app.lib import Batch
celery = init_celery(sentiment_app)

@celery.task()
def create_batch(quantity = 1):
  Batch.enqueue(quantity)

@celery.task()
def train():
	#pass
	#从redis 拿数据，放到trainer
	#batch 格式 [batch, labels]
	print(Batch.dequeue())
