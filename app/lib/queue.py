from app.config import redis
from .base import Base
import json

class Queue(Base):
  def __init__(self, name, namespace='sentiment'):
    self.key = '%s:%s' %(namespace, name)

  def pop(self, block=True, timeout = 5):
    if block:
        item = redis.blpop(self.key, timeout=timeout)
    else:
        item = redis.lpop(self.key)
    if item:
      item = item[1]  
    return json.loads(item)

  def push(self, item):
    return redis.rpush(self.key, item)

  def size(self):
    return redis.llen(self.key)