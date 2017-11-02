from flask_restful import Resource,reqparse,request
from ..lib import Vector
from task import update_redis

class VectorController(Resource):
  def get(self):
    update_redis()
    return {'Update redis:': 'Done'}



