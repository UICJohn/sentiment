from celery import Celery
from app.config import brokerUrl

celery_app = Celery('sentiment', broker = brokerUrl)

@celery_app.task
def train():
	print "hello world"