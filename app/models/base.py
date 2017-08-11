from flask_orator import Orator, jsonify

db_conn = Orator()

class Base(db_conn.Model):
	pass