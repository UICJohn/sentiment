from ..models import TrainingSet
from flask_restful import Resource
from ..lib import Batch, Trainer, TaskControl
from ..config import batchSize, redis
from task import init_worker, init_ps, create_batch
from celery import group

class TrainingTaskController(Resource):
  def post(self):
    TaskControl.clean_up_task()
    create_batch.delay()
    for i in range(0, 8):
      init_worker.apply_async(countdown = 2)
    return {"STATUS": "DONE"}