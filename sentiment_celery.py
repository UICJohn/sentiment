# from app.config import brokerUrl
# from celery import Celery

# def init_celery(app):
# 	celery = Celery('sentiment', broker = brokerUrl)
# 	celery.conf.update(app.config)
# 	TaskBase = celery.Task
# 	class ContextTask(TaskBase):
# 		abstract = True
# 		def __call__(self, *args, **kwargs):
# 			with app.app_context():
# 				return TaskBase.__call__(self, *args, **kwargs)
# 	celery.Task = ContextTask
# 	return celery
from app.config import celery_app
from app.lib import Batch

@celery_app.task()
def create_batch(quantity = 1):
	Batch.enqueue()
	return "Done"