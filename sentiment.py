import pdb
from app.config import *
from app.api import PredictionController
from app.api import TrainingSetController
from app.api import TrainingTaskController
from app.api import TestingController
from app import sentiment_app, sentiment_api

sentiment_api.add_resource(TrainingSetController, '/')
sentiment_api.add_resource(TrainingTaskController, '/training')
sentiment_api.add_resource(PredictionController, '/prediction')
sentiment_api.add_resource(TestingController, '/testing')

if __name__ == '__main__':
  redis.flushdb()
  sentiment_app.run(host='0.0.0.0')