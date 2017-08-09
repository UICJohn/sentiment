from .base import Base
from ..config import cache_db

class Cache(Base):
	# TODO due with cache
	@classmethod
	def set(self, key, value, expiry_time = 0):
		cache_db.setex(key, value, expiry_time)

	@classmethod
	def get(key):
		if cache_db.exist(key):
			return cache_db.get(key)
		else:
			return None
	@classmethod
	def incr(key, value = 1):
		cache_db.incr(key, value)

	@classmethod
	def clear(key):
		self.set(key, 0, 0)