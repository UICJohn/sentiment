from orator.seeds import Seeder
from os import listdir
from os.path import isfile, join
import numpy as np
import re
import json
from app.models import TrainingSet as ts
from app.models import EmMatrix as em

class TrainingSetTable(Seeder):
	def run(self):
		dataset_path = "aclImdb/train/"
		positiveFiles = [dataset_path + 'pos/' + f for f in listdir(dataset_path + "pos/") if isfile(join(dataset_path + 'pos/', f))]
		for fname in positiveFiles:
			self.loadFile(fname, 1)
		negativeFiles = [dataset_path + 'neg/' + f for f in listdir(dataset_path + "neg/") if isfile(join(dataset_path + 'neg/', f))]
		for fname in negativeFiles:
			self.loadFile(fname, -1)
	
	def word2Id(self, words):
		ids = []
		for i, word in enumerate(words.split(" ")):
			word = em.where("word", word).first()
			if(word):
				ids.append(word.id)
			else:
				ids.append(4000000)
		return json.dumps(ids)
		

	def loadFile(self, fname, label):
		with open(fname) as f:
			for line in f:
				words = self.stringClean(line)
				if not ts.where("words", words).first():
					ts.insert({ "words": words, "label": label, "word_ids": self.word2Id(words)})
					print("Trainning Set Count : " + str(ts.count()))

	def stringClean(self, string):
		special_chars = re.compile("[^A-Za-z0-9 ]+")
		string = string.lower().replace("<br />", "")
		return re.sub(special_chars, "", string)