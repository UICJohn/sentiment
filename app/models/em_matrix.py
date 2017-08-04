from .base import Base
class EmMatrix(Base):
	__casts__ = {'vector': 'array'}