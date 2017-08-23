from ..models import EmMatrix

def stringClean(string):
  special_chars = re.compile("[^A-Za-z0-9 ]+")
  string = string.lower().replace("<br />", "")
  return re.sub(special_chars, "", string)

def getMatrixByWordIds(word_ids, max_sentence_len):

  matrix = []

  for word_id in word_ids:
    word = EmMatrix.where('id', word_id).first()
    if word:
      matrix.append(word.vector)
    else:
      matrix.append([0] * 300)
  for l in range(len(matrix), max_sentence_len - 1):
    matrix.append([0] * 300)

  return matrix

def getMatrixBySentence(sentence, max_sentence_len):

  matrix = []
  word_ids = []

  words = stringClean(sentence).split()

  for word in words:

    word_vector = EmMatrix.where('word',word).first()

    if word_vector:
      word_ids.append(word_vector.id)
    else:
      word_ids.append(4000000)    

 
  matrix = getMatrixByWordIds(word_ids, max_sentence_len)

  return matrix

