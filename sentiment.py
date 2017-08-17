import pdb
from app.config import *
from app.api import TrainingSetController
from app.api import TrainingTaskController
from app import app, sentiment_api

sentiment_api.add_resource(TrainingSetController, '/')
sentiment_api.add_resource(TrainingTaskController, '/training')


if __name__ == '__main__':
	app.run()
