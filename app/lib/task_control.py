from ..config import redis_host, redis_port, redis
import json
import redis_lock

class TaskControl():
  def __init__(self, task_type = 'worker'):
    self.task_type = task_type

  def add_task(self):
    task_index = "0"
    with redis_lock.Lock(redis, self.task_type+"_lock", expire = 60):
      if redis.get(self.task_type + "_index"):
        task_index = redis.get(self.task_type + "_index")
      redis.incr(self.task_type + "_index")
    return int(task_index)