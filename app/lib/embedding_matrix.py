from ..config import embedding_redis
from ..models import TrainerSet, EmMatrix
class EmbeddingMatrix(Base):
  def add_vector(self, training_set):
    for word_id in training_set.word_ids:
      embedding_redis.hsetnx("embedding_matrix", word_id, pickle.dumps(EmMatrix.find_or_fail(word_id).vector))