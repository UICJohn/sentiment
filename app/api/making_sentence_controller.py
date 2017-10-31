from flask_restful import Resource,reqparse,request
from ..lib import Maker
from task import make



class MakingSentenceController(Resource):


  def post(self):
    numbers = request.values.get('Sentence number')
    make(numbers)
    return {'Making status:': 'Done'}