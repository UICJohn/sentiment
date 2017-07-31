from orator.seeds import Seeder
from models import EmMatrix as em
import os
import gensim

class InitialEmbeddingMatrix(Seeder):

	def run(self):
		mtx = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
		for key in mtx.vocab.keys():
			word2vector = em.insert({
				'word': key.encode("UTF-8"),
				'vector': self.convert2array(mtx.wv[key])
			})
			print("Key: " + key + " Done!")

	def convert2array(self, np_array):
		arr = []
		for num in np_array:
			arr.append(num.item())
		return arr
