from celery import Celery
from app.config import brokerUrl
from app.models import TrainingSet
from app.lib import Trainer
celery_app = Celery('sentiment', broker = brokerUrl)

@celery_app.task
def train(batchSize):
	trainer = Trainer(batchSize)
