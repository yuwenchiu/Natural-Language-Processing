# -*- coding: utf-8 -*-
import re, math
from collections import Counter, defaultdict
import nltk
from nltk.tokenize import TweetTokenizer
import numpy as np
import pandas as pd
import scipy.optimize as opt

def get_corpus(link):
  df = pd.read_json(link, lines=True)
  corpus = df.values.tolist()
  return corpus

def tokenize(d):
  tknzr = TweetTokenizer()
  tokens = tknzr.tokenize(d.lower())
  tokens = ['<s>'] + tokens + ['</s>']
  return tokens

"""#Part 1
Build a 2-gram model for the Twitter train data
"""

def perplexity(sentence):
  sent_bigrams = list(nltk.bigrams(sentence))
  probabilities = [(1 + counts[w1][w2])/(len(vocab) + sum(counts[w1].values())) for w1, w2 in sent_bigrams]
  N = len(sent_bigrams)
  cross_entropy =-1/N * sum([math.log(p, 2) for p in probabilities])
  return math.pow(2, cross_entropy)

"""Build forward model"""

train_corpus = get_corpus('http://bit.ly/nlp-tweet-train')
test_corpus = get_corpus('http://bit.ly/nlp-tweet-test')
train_bigram = []
vocab = Counter()
for item in train_corpus:
  tokens = tokenize(item[0])
  voc_tmp = Counter(tokens)
  vocab.update(voc_tmp)
  train_bigram.extend(nltk.bigrams(tokens))
counts = defaultdict(lambda: defaultdict(lambda: 0))
for w1, w2 in train_bigram:
  if vocab[w1] < 3:
    w1 = '<UNK>'
  if vocab[w2] < 3:
    w2 = '<UNK>'
  counts[w1][w2] += 1

"""Train data tweets avg. perplexity"""

sum_perplexity = 0
for item in train_corpus:
  tokens = tokenize(item[0])
  sum_perplexity += perplexity(tokens)
print('Average perplexity for Train data tweets: ')
print(sum_perplexity / len(train_corpus))

"""Test data tweets avg. perplexity"""

sum_perplexity = 0
for item in test_corpus:
  tokens = tokenize(item[0])
  sum_perplexity += perplexity(tokens)
print('Average perplexity for Test data tweets: ')
print(sum_perplexity / len(test_corpus))

"""#Part 2
Build a bi-directional 2-gram model by training on the Twitter train data
"""

def opt_perplexity(sentence, gamma):
  fwd_bigrams = list(nltk.bigrams(sentence))
  bwd_bigrams = list(nltk.bigrams(sentence[::-1]))
  probabilities = []
  for w1, w2 in fwd_bigrams:
    i = fwd_bigrams.index((w1, w2))
    if i == len(fwd_bigrams)-1:
      break
    fwd_prob = gamma*((1 + counts[w1][w2])/(len(vocab) + sum(counts[w1].values())))
    w3 = fwd_bigrams[i+1][1]
    w4 = fwd_bigrams[i+1][0]
    bwd_prob = (1-gamma)*((1 + counts_back[w3][w4])/(len(vocab) + sum(counts_back[w3].values())))
    probabilities.append(fwd_prob + bwd_prob)
  N = len(fwd_bigrams)
  cross_entropy =-1/N * sum([math.log(p, 2) for p in probabilities])
  return math.pow(2, cross_entropy)

"""Build backward model"""

train_bigram_back = []
for item in train_corpus[::-1]:
  tokens = tokenize(item[0])
  train_bigram_back.extend(nltk.bigrams(tokens[::-1]))
counts_back= defaultdict(lambda: defaultdict(lambda: 0))
for w1, w2 in train_bigram_back:
  if vocab[w1] < 3:
    w1 = '<UNK>'
  if vocab[w2] < 3:
    w2 = '<UNK>'
  counts_back[w1][w2] += 1

"""Train data tweets avg. perplexity at the optimal 𝛾"""

sum_perplexity = 0
for i in np.arange(0.0, 1.0, 0.05):
  print(i)
  tmp_opt = 0
  for item in train_corpus:
    tokens = tokenize(item[0])
    tmp_opt += opt_perplexity(tokens, i)
  if sum_perplexity == 0 or tmp_opt < sum_perplexity:
    sum_perplexity = tmp_opt
print('Average perplexity for Train data tweets at the optimal 𝛾: ', end = '')
print(sum_perplexity / len(train_corpus))

"""Test data tweets avg. perplexity at the optimal 𝛾"""

sum_perplexity = 0
opt_gamma = 0.00
for i in np.arange(0.0, 1.0, 0.05):
  tmp_opt = 0
  for item in test_corpus:
    tokens = tokenize(item[0])
    tmp_opt += opt_perplexity(tokens, i)
  if sum_perplexity == 0 or tmp_opt < sum_perplexity:
    sum_perplexity = tmp_opt
    opt_gamma = i
print('𝛾 that minimizes the perplexity of the Twitter test data:', end = '')
print(opt_gamma)
print('Average perplexity for Test data tweets at the optimal 𝛾: ', end = '')
print(sum_perplexity / len(test_corpus))