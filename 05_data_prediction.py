# -*- coding: utf-8 -*-
from google.colab import files
train_file = files.upload()
test_file = files.upload()

import io
import pandas as pd
from collections import Counter
import nltk

from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter
import json
import requests
import heapq
import time

train_data = pd.read_json(io.BytesIO(train_file['train_gold.json']), lines=True)
test_data = pd.read_json(io.BytesIO(test_file['dev_unlabeled.json']), lines=True)

"""#Part 1"""

co_10 = Counter()
s=","
Ncategories=[0]*6
categories_c=Counter()
for i in train_data.categories.to_list():
  for j in list(nltk.ngrams(i,2)):
    co_10.update([s.join(j)])
  Ncategories[len(i)-1]+=1
  categories_c.update(i)

print("Number of samples that have N categories:")
for index,count in enumerate(Ncategories):
  print(str(index+1)+": "+str(count))

print("Category distribution:")
print(categories_c.most_common(43) )

print("10 most common pairs of co-occurring categories:")
print(co_10.most_common(10))

"""#Part 2

Majority Prediction
"""

majority_six = []
for label in categories_c.most_common(6):
  majority_six.append(label[0])
with open('dev.json', 'w', encoding='utf-8') as f:
  for line in test_data.values.tolist(): 
    ans = {"idx": line[0] ,"categories": majority_six, "reply": line[2], "text": line[1]}
    f.write(json.dumps(ans, ensure_ascii=False))
    f.write("\n") 

time.sleep(30)

files.download('dev.json')

"""Naive Bayes"""

vectorizer = TfidfVectorizer()
vectorizer.fit(train_data['text'])
train_X = vectorizer.transform(train_data['text'])
test_X = vectorizer.transform(test_data['text'])
train_Y = [categories[0] for categories in train_data['categories'].to_list()]

model = MultinomialNB()
model.fit(train_X.toarray(),train_Y)
pred_Y = model.predict(test_X.toarray())

probabilities = model.predict_proba(test_X.toarray())
catagories = ["agree", "applause", "awww", "dance", "deal_with_it", "do_not_want", "eww", "eye_roll", "facepalm", "fist_bump", "good_luck", "happy_dance", "hearts", "high_five", "hug", "idk", "kiss", "mic_drop", "no", "oh_snap", "ok", "omg", "oops", "please", "popcorn", "scared", "seriously", "shocked", "shrug", "sigh", "slow_clap", "smh", "sorry", "thank_you", "thumbs_down", "thumbs_up", "want", "win", "wink", "yawn", "yes", "yolo", "you_got_this"]

cnt = 0
with open('dev.json', 'w', encoding='utf-8') as f:
  for line in test_data.values.tolist(): 
    max_six_idx = list(map(list(probabilities[cnt]).index, heapq.nlargest(6, list(probabilities[cnt]))))
    max_six = []
    for i in max_six_idx:
      max_six.append(catagories[i])
    ans = {"idx": line[0] ,"categories": max_six, "reply": line[2], "text": line[1]}
    f.write(json.dumps(ans, ensure_ascii=False))
    f.write("\n")
    cnt += 1

time.sleep(30)

files.download('dev.json')