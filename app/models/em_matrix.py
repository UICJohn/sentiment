from orator import Model
# from application import db

class EmMatrix(db.Model):
	__casts__ = {'vector': 'array'}