from flask_restful import Resource,reqparse,request
from ..lib import Tester
from task import test


class TestingController(Resource):
  def post(self):
    # result = test()
    test()
    return {'test data size':'None', 'correct':'None', 'incorrect':'None', 'accuracy':'None'}
