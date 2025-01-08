import torch
import torch.optim as optim

import numpy as np

import string

import nltk
from nltk import word_tokenize

from tqdm import tqdm
import random

# Skip-gram
class Word2Vec():
    def __init__(self, window=5, embedding_dim=100, alpha=0.01, epochs=10):
        self.window = window
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.epochs = epochs

    # optional, can be done by users in their own way
    def data_preprocessing(self, list_of_texts):
        texts = [text.lower() for text in list_of_texts]
        texts_by_sentences = [nltk.sent_tokenize(text) for text in texts]
        sentences = [sentence for text in texts_by_sentences for sentence in text]
        tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
        return tokens

    def _initialize_vocabulary(self, data):
        self.vocabulary = sorted(list(set([w for sentence in data for w in sentence])))
        self.embeddings = {word: torch.tensor(np.random.uniform(-1, 1, (self.embedding_dim)), dtype=torch.float32, requires_grad=True) for word in self.vocabulary}

    def _get_window_sets(self, data):
        res = []
        for sentence in data:
            for center_word_i, center_word in enumerate(sentence):
                window_start = max(0, center_word_i-self.window)
                window_end = min(len(sentence), center_word_i+self.window+1)
                window_words = sentence[window_start:window_end]
                context = [word for word_i, word in enumerate(window_words) if word_i+window_start != center_word_i]
                res.append((center_word, context))
        return res

    def _parameters(self):
        return list(self.embeddings.values())

    def _negative_sampling(self, center_word, context_words):
        words_left = list(set(self.vocabulary) - set(context_words) - set([center_word]))
        return [random.choice(words_left) for i in range(self.window)] # let number of negative words be equal to window size

    def _loss(self, center_tensor, context_tensors, neg_tensors):
        pos_scores = torch.matmul(context_tensors, center_tensor)
        pos_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores)))

        neg_scores = torch.matmul(neg_tensors, center_tensor)
        neg_loss = -torch.mean(torch.log(torch.sigmoid(-neg_scores)))

        return pos_loss + neg_loss

    # data = [[sentence1_word1, sentence1_word2, ...], [sentence2_word1, ...], ..., [sentence_n_word1, ...]]
    def train(self, data, update=False):
        if not update:
            self._initialize_vocabulary(data)

        # window_sets = [(center_word1, [context_word1, context_word2, ...]), ...] flatten for all the sentences
        window_sets = self._get_window_sets(data)

        optimizer = optim.Adam(self._parameters(), lr=self.alpha)

        for epoch in range(self.epochs):
            total_loss = 0
            np.random.shuffle(window_sets)
            for center_word, context_words in tqdm(window_sets):
                negative_words = self._negative_sampling(center_word, context_words)

                # for update=True there can be new words
                if update:
                    if center_word not in self.embeddings or any([word not in self.embeddings for word in context_words]):
                        continue

                center_tensor = self.embeddings[center_word]
                context_tensors = torch.stack([self.embeddings[word] for word in context_words])
                neg_tensors = torch.stack([self.embeddings[word] for word in negative_words])

                optimizer.zero_grad()
                loss = self._loss(center_tensor, context_tensors, neg_tensors)
                loss.backward()
                optimizer.step()

                total_loss += loss

            print(f"{epoch} done; loss = {total_loss}")

    def get_word_embedding(self, word):
        emb = self.embeddings.get(word, None)
        if emb is not None:
            emb = emb.detach().numpy()
        return emb

    def _cosine_diff(self, emb1, emb2):
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1)*np.linalg.norm(emb2))

    def words_similarity(self, word1, word2):
        if not word1 in self.vocabulary:
            print(f"word '{word1}' not found in vocabulary, similarity cannot be computed")
            return
        if not word2 in self.vocabulary:
            print(f"word '{word2}' not found in vocabulary, similarity cannot be computed")
            return

        emb1 = self.embeddings[word1].detach().numpy()
        emb2 = self.embeddings[word2].detach().numpy()

        return self._cosine_diff(emb1, emb2)

    def get_closest(self, word, n=5):
        if not word in self.vocabulary:
            print(f"Word {word} not found in vocabulary")
            return

        emb = self.embeddings[word].detach().numpy()
        sim = {w: self._cosine_diff(emb, self.embeddings[w].detach().numpy()) for w in self.vocabulary if w != word}
        sim = sorted(sim.items(), key=lambda x: x[1], reverse=True)
        res = [x for x in sim[:n]]
        return res
