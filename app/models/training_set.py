from .base import Base
from ..config import redis
from .. import db_conn
class TrainingSet(Base):
  __casts__={
  'word_ids': 'list'
  }

  @classmethod
  def maxSentenceLen(cls):
    max_sentence_len = redis.get("max_sentence_len")
    if max_sentence_len:
      return int(max_sentence_len)
    else:
      max_sentence_len = cls.select(db_conn.raw("max(array_length(regexp_split_to_array(words, '\s|\t'), 1)) as max_len")).first().max_len
      redis.set("max_sentence_len", max_sentence_len)
      return max_sentence_len