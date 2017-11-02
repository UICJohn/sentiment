from flask_restful import Resource
from os import listdir
from os.path import isfile, join

import numpy as np
import re
import os
import random
from datetime import datetime

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
    s1, s2,s3 = self.__recombinate(subject_list,verb_list,object_list)
    # if len(object_list[0]) > 100:
    # print("output files is ", len(s1), " s2 :", len(s2), " s3 :", len(s3), len(s1)+ len(s2)+len(s3))
    # num_p = self.countFiles('/Users/chih/Documents/IOS/sentiment_dev/sentiment/sentence/tmp_pos/')
    # num_n = self.countFiles('/Users/chih/Documents/IOS/sentiment_dev/sentiment/sentence/tmp_neg/')
    # print("---------------", num_p + num_n)


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
    dt = datetime.now()
    s1 = []
    s2 = []
    s3 = []
    # print("----------------------------------",len(object_list[0]), len(object_list[1]) )
    for i in range(0,len(subject_list)):
      if i== 0:
        for j in range(0,len(subject_list[i])):
          for k in range(0, len(verb_list[i])):
            sentence = subject_list[i][j] + " " + verb_list[0][k]
            if k > 1:
              for p in range(0, len(object_list[0])):
                #not negative -> positive
                s1.append(sentence + " " + object_list[0][p])
                fh = '/Users/chih/Documents/IOS/sentiment_dev/sentiment/sentence/tmp_pos/'+ str(i)+ str(j)+ str(k) + str(p) + str(0) +'.txt'
                with open(fh, "w") as txt:
                  txt.write(sentence + " " + object_list[0][p])
              for q in range(0,len(object_list[1])):
                #not positive ->negative
                s1.append(sentence + " " + object_list[1][q])
                fh = '/Users/chih/Documents/IOS/sentiment_dev/sentiment/sentence/tmp_neg/'+  str(i)+ str(j)+ str(k) + str(q) +str(1) +'.txt'
                with open(fh, "w") as txt:
                  txt.write(sentence + " " + object_list[1][q])                
                
            else:
              for p in range(0,len(object_list[0])):
                #negative ->negative
                s1.append(sentence + " " + object_list[0][p])
                fh = '/Users/chih/Documents/IOS/sentiment_dev/sentiment/sentence/tmp_neg/'+  str(i)+ str(j)+ str(k)+ str(p)+str(0) +'.txt'
                with open(fh,"w") as txt:
                  txt.write(sentence + " " + object_list[0][p])
              for q in range(0,len(object_list[1])):
                #positive -> positive
                s1.append(sentence + " " + object_list[1][q])
                fh = '/Users/chih/Documents/IOS/sentiment_dev/sentiment/sentence/tmp_pos/'+  str(i)+ str(j) + str(k) + str(q) +str(1) +'.txt'
                with open(fh, "w") as txt:
                  txt.write(sentence + " " + object_list[1][q])
        # print("The line 99 is +++++++++++++++++++++++ ", self.countFiles('/Users/chih/Documents/IOS/sentiment_dev/sentiment/sentence/tmp_pos/') + self.countFiles('/Users/chih/Documents/IOS/sentiment_dev/sentiment/sentence/tmp_neg/'))
      elif i == 1:
        for j in range(0,len(subject_list[i])):
          for k in range(0, len(verb_list[i])):
            sentence = subject_list[i][j]+ " " +verb_list[1][k]
            # #print('-----------------------', sentence)
            if k > 1:
              for p in range(0,len(object_list[0])):
                #not negative -> positive
                s2.append(sentence + " " + object_list[0][p])
                fh = '/Users/chih/Documents/IOS/sentiment_dev/sentiment/sentence/tmp_pos/'+ str(i)+ str(j) + str(k) + str(p) + str(0) +'.txt'
                with open(fh, "w") as txt:
                  txt.write(sentence + " " + object_list[0][p])
              for q in range(0, len(object_list[1])):
                #not positive -> negative
                s2.append(sentence + " " + object_list[1][q])
                fh = '/Users/chih/Documents/IOS/sentiment_dev/sentiment/sentence/tmp_neg/'+ str(i)+ str(j) + str(k) + str(q) + str(1) +'.txt'
                with open(fh, "w") as txt:
                  txt.write(sentence + " " + object_list[1][q])
            else:
              for p in range(0,len(object_list[0])):
                #negative -> negative
                s2.append(sentence + " " + object_list[0][p])
                fh = '/Users/chih/Documents/IOS/sentiment_dev/sentiment/sentence/tmp_neg/'+  str(i)+ str(j)+ str(k)+ str(p)+str(0) +'.txt'
                with open(fh,"w") as txt:
                  txt.write(sentence + " " + object_list[0][p])
              for q in range(0,len(object_list[1])):
                #positive -> positive
                s2.append(sentence + " " + object_list[1][q])
                fh = '/Users/chih/Documents/IOS/sentiment_dev/sentiment/sentence/tmp_pos/'+  str(i)+ str(j) +str(k)+ str(q) +str(1) +'.txt'
                with open(fh, "w") as txt:
                  txt.write(sentence + " " + object_list[1][q])
        # print("The line 131 is +++++++++++++++++++++++ ", self.countFiles('/Users/chih/Documents/IOS/sentiment_dev/sentiment/sentence/tmp_pos/') + self.countFiles('/Users/chih/Documents/IOS/sentiment_dev/sentiment/sentence/tmp_neg/'))

      elif i == 2:
        for j in range(0,len(subject_list[i])):
          for k in range(0, len(verb_list[i])):
            sentence = subject_list[i][j]+ " " +verb_list[2][k]
            #print('-----------------------', sentence)
            if k > 1:
              for p in range(0,len(object_list[0])):
                #not negative -> positive
                s3.append(sentence + " " + object_list[0][p])
                fh = '/Users/chih/Documents/IOS/sentiment_dev/sentiment/sentence/tmp_pos/'+ str(i)+ str(j) + str(k) + str(p) + str(0) +'.txt'
                with open(fh, "w") as txt:
                  txt.write(sentence + " " + object_list[0][p])
              for q in range(0,len(object_list[1])):
                #not positive -> negative
                s3.append(sentence + " " + object_list[1][q])
                fh = '/Users/chih/Documents/IOS/sentiment_dev/sentiment/sentence/tmp_neg/'+ str(i)+ str(j) + str(k) + str(q) + str(1) +'.txt'
                with open(fh, "w") as txt:
                  txt.write(sentence + " " + object_list[1][q])
            else:
              for p in range(0,len(object_list[0])):
                #negative -> negative
                s3.append(sentence + " " + object_list[0][p])
                fh = '/Users/chih/Documents/IOS/sentiment_dev/sentiment/sentence/tmp_neg/'+  str(i)+ str(j)+ str(k)+ str(p)+str(0) +'.txt'
                with open(fh, "w") as txt:
                  txt.write(sentence + " " + object_list[0][p])
              for q in range(0,len(object_list[1])):
                #positive -> positive
                s3.append(sentence + " " + object_list[1][q])
                fh = '/Users/chih/Documents/IOS/sentiment_dev/sentiment/sentence/tmp_pos/'+  str(i)+ str(j) +str(k)+ str(q) +str(1) +'.txt'
                with open(fh, "w") as txt:
                  txt.write(sentence + " " + object_list[1][q])             
        # print("The line 164 is +++++++++++++++++++++++ ", self.countFiles('/Users/chih/Documents/IOS/sentiment_dev/sentiment/sentence/tmp_pos/') + self.countFiles('/Users/chih/Documents/IOS/sentiment_dev/sentiment/sentence/tmp_neg/'))

    return s1,s2,s3          

  @classmethod
  def countFiles(self, filePath):
    #convert generated sentences into .txt
    files = [filePath + f for f in listdir(filePath) if isfile(join(filePath, f)) and not f.startswith('.')]
    num = 0
    for f in files:
      num = num + 1;
      # print("Files number is =================== ", num)
    return num










      