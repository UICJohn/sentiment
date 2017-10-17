from ..config import redis
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

  def get_task_index(self):
    task_index = 0
    with redis_lock.Lock(redis, self.task_type+"_lock", expire = 60):
      if redis.get(self.task_type + "_index"):
        task_index = int(redis.get(self.task_type + "_index"))
    return task_index

  @classmethod
  def clean_up_task(cls):
    redis.set("worker_index", 0)
    redis.set("ps_index", 0)