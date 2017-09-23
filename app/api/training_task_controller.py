from ..models import TrainingSet
from flask_restful import Resource
from ..lib import Batch, Trainer, TaskControl
from ..config import batchSize, redis
from task import init_worker, init_ps
from celery import group

class TrainingTaskController(Resource):
  def post(self):
    TrainingSet.where('iterations', '!=', 1).update(iterations = 1)
    for i in range(0, 8):
      init_worker.apply_async(countdown = 10*i)
    return {"STATUS": "DONE"}