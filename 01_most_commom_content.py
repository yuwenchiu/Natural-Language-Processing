# -*- coding: utf-8 -*-
import re
import urllib
import math
import pandas as pd
from collections import Counter

"""#Fetching the Corpus
`get_corpus()` reads the CSV file, and then return a list of the news headlines
"""

def get_corpus():
  df = pd.read_csv('https://raw.githubusercontent.com/bshmueli/108-nlp/master/reuters.csv')
  print("Dataset columns", df.columns)
  print("Dataset size", len(df))
  corpus = df.content.to_list()
  return corpus

def get_title():
  df = pd.read_csv('https://raw.githubusercontent.com/bshmueli/108-nlp/master/reuters.csv')
  title_list = df.title.to_list()
  return title_list

def get_stopwords():
  f = urllib.request.urlopen(r'https://raw.githubusercontent.com/bshmueli/108-nlp/master/stopwords.txt')
  stopword = []
  for line in f:
    decode_line = line.decode("utf-8")
    decode_line = decode_line.split("\n")
    stopword.append(decode_line[0])
  return stopword

def tokenize(document):
  words_lower = document.lower()
  words_split = re.split(r'\W+', words_lower)
  stopwords = get_stopwords()
  words = words_split.copy()
  for w in words_split:
    for sw in stopwords:
      if w == sw or w =="":
        words.remove(w)
        break
  return words

"""#Computing word frequencies
`get_vocab(corpus)` computes the word frequencies in a given corpus. It returns a list of 2-tuples. Each tuple contains the token and its frequency.
"""

def get_vocab(corpus):
  vocabulary = Counter()
  for document in corpus:
    tokens = tokenize(document)
    tokenize_doc.append(tokens)
    vocabulary.update(tokens)
  return vocabulary

"""#Compute TF-IDF Vector
`doc2vec(doc)` returns a TF-IDF vector for document `doc`, corresponding to the presence of a word in `vocab`
"""

def tf_idf(word, tf):
  df = 0
  sum = 0
  for doc in tokenize_doc:
    if word in doc:
      df += 1
  return tf * math.log(len(corpus) / df)

def doc2vec(doc):
  tokenize_doc = tokenize(doc)
  vec = []
  for token in vocab:
    if token[0] in tokenize_doc:
      vec.append(tf_idf(token[0], tokenize_doc.count(token[0])))
    else: vec.append(0)
  return vec

"""Cosine similarity between two numerical vectors"""

def cosine_similarity(vec_a, vec_b):
  assert len(vec_a) == len(vec_b)
  if sum(vec_a) == 0 or sum(vec_b) == 0:
    return 0 # hack
  a_b = sum(i[0] * i[1] for i in zip(vec_a, vec_b))
  a_2 = sum([i*i for i in vec_a])
  b_2 = sum([i*i for i in vec_b])
  return a_b/(math.sqrt(a_2) * math.sqrt(b_2))

def doc_similarity(doc2vec_a, doc_b, id):
  print(id)
  return cosine_similarity(doc2vec_a, doc2vec(doc_b))

"""# Find Similar Documents
Find and print the $k$ most similar titles to a given title
"""

def k_similar(seed_id, k):
  title_list = get_title()
  seed_doc = corpus[seed_id]
  print('> "{}"'.format(title_list[seed_id]))
  doc_a = doc2vec(seed_doc)
  similarities = [doc_similarity(doc_a, doc, id) for id, doc in enumerate(corpus)]
  top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i])[-k:]
  nearest = [[title_list[id], similarities[id]] for id in top_indices]
  print()
  for story in reversed(nearest):
    print('* "{}" ({})'.format(story[0], story[1]))

"""# Test our program"""

corpus = get_corpus()
tokenize_doc = []
vocab = get_vocab(corpus).most_common(1000)
k_similar(546, 5)
