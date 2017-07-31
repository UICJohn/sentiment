from orator import Model
from app import db

class EmMatrix(db.Model):
	__casts__ = {'vector': 'array'}