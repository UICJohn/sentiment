from flask_restful import Resource
from os import listdir
from os.path import isfile, join

import numpy as np
import re
import os
import random



class Maker():
  """docstring for Maker"""
  @classmethod
  def process(self, sentence_num):
    subject_list = self.readFile('/Users/chih/Documents/IOS/sentiment_dev/sentiment/sentence/Subject/')
    verb_list = self.readFile('/Users/chih/Documents/IOS/sentiment_dev/sentiment/sentence/Verb/')
    object_list = self.readFile('/Users/chih/Documents/IOS/sentiment_dev/sentiment/sentence/Object/')


    # print("-------------------------", subject_list)
    # print("++++++++++++++++++++++++ ", verb_list)
    # print("++++++++++++++++++++++++ ", object_list)
    self.__recombinate(subject_list,verb_list,object_list)
    # if len(object_list[0]) > 100:

      
    # else:


  @classmethod
  def readFile(self, filePath):
    files = [filePath + f for f in listdir(filePath) if isfile(join(filePath, f)) and not f.startswith('.')]
    word_list = []

    for f in files:
      print(f)
      with open(f, "r", encoding = 'utf-8') as f:
        #print(f.readline())
        # print(self.__cleanSentences(f.readline()))
        line = f.readline()
        line = self.__cleanSentences(line)
        split = line.split(',')
        # print(split)
        word_list.append(split)
    return word_list



  @classmethod
  def __cleanSentences(self, string):
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    # string = re.sub(strip_special_chars, "", string.lower())
    print('clean sententce ', string)
    return string
  @classmethod
  def __recombinate(self, subject_list, verb_list, object_list):
    for i in range(0,len(subject_list)):
      if i== 0:
        for j in range(0,len(subject_list[i])):
          for k in range(0, len(verb_list[i])):
            sentence = subject_list[i][j] + " " + verb_list[0][k]
            print('-----------------------', sentence)
            for p in range(0, len(object_list[0])):
              sentence = sentence + " " + object_list[0][p]

      elif i == 1:
        for j in range(0,len(subject_list[i])):
          for k in range(0, len(verb_list[i])):
            sentence = subject_list[i][j]+ " " +verb_list[1][k]
            print('-----------------------', sentence)
            for p in range(0, len(object_list[0])):
              sentence = sentence + " " + object_list[0][p]            
      elif i == 2:
        for j in range(0,len(subject_list[i])):
          for k in range(0, len(verb_list[i])):
            sentence = subject_list[i][j]+ " " +verb_list[2][k]
            print('-----------------------', sentence)
            for p in range(0, len(object_list[0])):
              sentence = sentence + " " + object_list[0][p]












      