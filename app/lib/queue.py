from app.config import redis, redis_locker
import asyncio
import pdb
from .base import Base
import json

class Queue(Base):
  def __init__(self, name, namespace='sentiment', max_count = None):
    self.key = '%s:%s' %(namespace, name)
    self.max_count = max_count

  def pop(self, block=True, timeout = 5):
    if block:
        item = redis.blpop(self.key, timeout=timeout)
    else:
        item = redis.lpop(self.key)
    if item:
      item = json.loads(item[1].decode('utf-8'))
    return item

  def push(self, item):
    if (self.max_count):
      pdb.set_trace()
      lock = redis_locker.lock(self.key, 1000)
      if (lock):
        if (self.max_count and (self.size() < self.max_count)):
          res = redis.rpush(self.key, item)
          redis_locker.unlock(lock)
          return res
      return False
    else:
      return redis.rpush(self.key, item)

  def size(self):
    return redis.llen(self.key)