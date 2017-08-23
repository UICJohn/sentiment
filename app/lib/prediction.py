from ..models import EmMatrix as em
from .base import Base
from ..config import batchSize, numClasses, numDimensions, redis
from ..models import TrainingSet as ts

class Prediction(Base):
  @classmethod
  def process(cls, sentence=None):

    return False