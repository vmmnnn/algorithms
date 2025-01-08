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

w2v = Word2Vec(epochs=3, embedding_dim=100)

preprocessed = w2v.data_preprocessing(list_of_texts)
print(f"preprocessing done; len(preprocessed)={len(preprocessed)}")

w2v.train(preprocessed)
print(f"training done")

res_filename = 'word2vec_model.pkl'
with open(res_filename, 'wb') as f:
    pickle.dump(w2v, f)
print(f"model saved to {res_filename}")
