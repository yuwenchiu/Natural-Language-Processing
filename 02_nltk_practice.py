# -*- coding: utf-8 -*-
import nltk
import pandas as pd
from unidecode import unidecode
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from nltk import ngrams
from nltk import pos_tag
nltk.download('averaged_perceptron_tagger')
from collections import Counter
import spacy
nlp = spacy.load("en_core_web_sm")
import re, math
from sklearn.metrics.pairwise import cosine_similarity

def get_corpus(link):
  df = pd.read_csv(link)
  corpus = df.content.to_list()
  title_list = df.title.to_list()
  return corpus, title_list

"""#Part 1
5 most frequent 2-grams
"""

def twograms_and_tags(document):
    uni_doc = unidecode(document)
    tokens = word_tokenize(uni_doc)
    return ngrams(tokens, 2), pos_tag(tokens)

def calculateNNP():
  for i in twogram:
    flag1 = False
    flag2 = False
    for j in tags:
      if (i[0] == j[0]) and (j[1] == 'NNP' or j[1] == 'NNPS'):
        flag1 = True
      if (i[1] == j[0]) and (j[1] == 'NNP' or j[1] == 'NNPS'):
        flag2 = True
      if flag1 is True and flag2 is True:
        freq.update([i])
        break

corpus1, title1 = get_corpus('https://raw.githubusercontent.com/bshmueli/108-nlp/master/reuters.csv')
freq = Counter()
for doc in corpus1:
  twogram, tags = twograms_and_tags(doc)
  calculateNNP()
print(freq.most_common(5))

"""#Part 2
5 most similar articles
"""

def remove_stop_and_punc(document):
  tokens = []
  for token in document:
    if (token.is_stop is False) and (token.pos_ is not 'PUNCT') and (token.pos_ is not 'SPACE'):
      tokens.append(token.lemma_.lower()+"_"+token.pos_)
  return tokens

def get_vocab(corpus):
  vocabulary = Counter()
  for document in corpus:
    vocabulary.update(document)
  return vocabulary

def tfidf_vectors(corpus, vocab):

  # return the TF-IDF vector for a single document
  def tfidf_vec(doc):
    doc_freqs = Counter(doc)
    return [doc_freqs[token] * math.log(N / df[token]) for token, _ in vocab]

  # first, compute the document frequency; df[token] is the number of documents containing token
  df = Counter()
  token_set = set([token for token, _ in vocab])
  for document in corpus:
    df.update(list(set(document) & token_set))

  N = len(corpus)
  # use df to compute the df-idf vector for each document

  return [tfidf_vec(doc) for doc in corpus]

corpus2, title2 = get_corpus('https://raw.githubusercontent.com/bshmueli/108-nlp/master/buzzfeed.csv')
combine_token = []
for doc in corpus2:
  doc = nlp(str(doc))
  combine_token.append(remove_stop_and_punc(doc))
vocabs = get_vocab(combine_token).most_common(512)
tfidf = tfidf_vectors(combine_token, vocabs)
similarity = cosine_similarity(tfidf)
top_indices = sorted(range(len(similarity[546])), key=lambda i: similarity[546][i])[-5:]
nearest = [[title2[id], similarity[546][id]] for id in top_indices]
print('> "{}"'.format(title2[546]))
print()
for story in reversed(nearest):
  print('* "{}" ({})'.format(story[0], story[1]))
