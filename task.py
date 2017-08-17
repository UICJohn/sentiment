from app import init_celery, sentiment_app
from app.lib import Batch
celery = init_celery(sentiment_app)

@celery.task()
def create_batch(quantity = 1):
    Batch.enqueue(quantity)