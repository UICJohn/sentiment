from app import init_celery, sentiment_app
from app.lib import Batch
from app.lib import Trainer
from app.config import GPU_NUMS
from kombu.common import Broadcast
from app.lib import Prediction

celery = init_celery(sentiment_app)
celery.conf.task_routes = {
  'tasks.init_worker': {
    'queue': 'worker_task',
    'exchange': 'worker_task'
  },
  'tasks.init_ps': {
    'queue': 'ps_task',
    'exchange': 'ps_task'
  },
  'tasks.create_batch':{
    'queue': 'create_batch',
    'exchange': 'create_batch'
  }
}

@celery.task(queue = "worker_tasks")
def init_worker():
  Trainer(task_type= 'worker').process()

@celery.task(queue = "ps_tasks")
def init_ps():
  Trainer(task_type= 'ps').process()

@celery.task(queue = 'worker_tasks')
def create_batch():
	Batch.enqueue()
 
@celery.task()
def output(sentence):
  print("############### Starting run predict", sentence)
  
  label = Prediction.process(sentence)
  return label
