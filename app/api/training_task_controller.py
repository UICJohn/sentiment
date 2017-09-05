from flask_restful import Resource
from ..lib import Batch, Trainer
from ..config import batchSize, redis
from task import init_worker, init_ps

class TrainingTaskController(Resource):
  def post(self):
    # for i in range(1):
    init_worker.apply_async(countdown = 5)
    # t = Trainer()
    # t.process()
    return {"STATUS": "DONE"}