from .base import Base
from ..config import redis
class TrainingSet(Base):
  __casts__={
  'word_ids': 'list'
  }
  @classmethod
  def maxVectorSize(cls):
    max_vector_size = redis.get("max_vector_size")
    if max_vector_size:
      return max_vector_size
    else:
      max_vector_size = cls.select(db_conn.raw("max(array_length(regexp_split_to_array(words, '\s|\t'), 1)) as max_len")).first().max_len
      redis.put("max_vector_size", max_vector_size, 180)
      return max_vector_size