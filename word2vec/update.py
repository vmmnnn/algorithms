#!/usr/bin/env python

import os
import pickle

from Word2Vec import Word2Vec

list_of_texts = []
path = "data/"
for filename in os.listdir(path):
    with open(f"{path}{filename}", 'r') as f:
        try:
            text = f.read()
            list_of_texts.append(text)
        except:
            print(f"{filename} skipped")
print(f"{len(list_of_texts)} texts read from {path}")


with open('word2vec_model.pkl', 'rb') as f:
    w2v = pickle.load(f)

w2v.epochs = 1 # one more

preprocessed = w2v.data_preprocessing(list_of_texts)
print(f"preprocessing done; len(preprocessed)={len(preprocessed)}")

w2v.train(preprocessed, update=True)
print(f"training done")

res_filename = 'word2vec_model_2.pkl'
with open(res_filename, 'wb') as f:
    pickle.dump(w2v, f)
print(f"model saved to {res_filename}")
