from celery import Celery
from app.config import brokerUrl
from app.models import TrainingSet
from app.lib import Batch
from app.config import redis, maxBatchCount

celery_app = Celery('sentiment', broker = brokerUrl)

@celery_app.task
def create_batch(batchSize):
  # if batch_count and batch_count < maxBatchCount:
  pass