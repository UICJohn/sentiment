from .database import database

ORATOR_DATABASES = database

DEBUG = True

batchSize = 24

maxBatchCount = 200

brokerUrl = 'redis://192.168.0.6:6379/0'