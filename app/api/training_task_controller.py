from flask_restful import Resource
from ..lib import Batch, Trainer
from ..config import batchSize, redis
from task import init_worker, init_ps

class TrainingTaskController(Resource):
  def post(self):
    init_ps.apply_async(countdown = 5)
    init_worker.apply_async(countdown = 5)
    return {"STATUS": "DONE"}