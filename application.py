import pdb
from app.config import *
from app.api import TrainingSetController
from app.api import TrainingTaskController

api.add_resource(TrainingSetController, '/')
api.add_resource(TrainingTaskController, '/training')


if __name__ == '__main__':
	app.run()
