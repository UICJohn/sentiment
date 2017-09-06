from flask_restful import Resource
from ..lib import Batch, Trainer
from ..config import batchSize, redis
from task import init_worker, init_ps
from celery import group

class TrainingTaskController(Resource):
  def post(self):
    init_ps.apply_async(countdown = 5)
    worker_tasks = group(init_worker.apply_async(countdown = 5) for i in range(0, 8))()
    worker_tasks.apply_async()
    return {"STATUS": "DONE"}