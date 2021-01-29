import sys
import argparse
import numpy
import pickle
import collections
from sklearn import preprocessing

## A Class Representing a Word Embedding
class word_embedding:

    ## Initializes a word_embedding object 
    def __init__(self, vecs, vocab, normalize=True, **kwargs):
        self.vecs = vecs
        self.dimension = self.vecs.shape[1]
        self.vocab = vocab
        self.vocab_index = {w:i for i,w in enumerate(self.vocab)}
        #if normalize:
         #   self.normalize()

    ## Retrieves an embedding given a word if it is in the vocabulary 
    def __getitem__(self, key):
        if not (key in self.vocab_index):
            print("Not in Vocabulary: ", key)
            raise KeyError
        else:
            return self.represent(key)
            
    ## Iterator through the vocabulary 
    def __iter__(self):
        return self.vocab_index.__iter__()
    
    ## Checks if in the vocabulary 
    def __contains__(self, key):
        return not (key in self.vocab_index)

    ## Loads in the vectors and corresponding vocab given the file path 
    @classmethod
    def load_vector(_class, path, normalize=True, add_context=False, **kwargs):
        vecs = numpy.load(path + "-w.npy", mmap_mode="c")
        with open(path + "-vocab.pkl", "rb") as fp:
            vocab = pickle.load(fp)
        return _class(vecs, vocab, normalize)

    ## Normalizes vectors
    def normalize(self):
        preprocessing.normalize(self.vecs, copy=False)

    ## Computes the cosine similarity of two words
    def similarity(self, w1, w2):
        v1 = self.represent(w1)
        v2 = self.represent(w2)
        if numpy.linalg.norm(v1) == 0 or numpy.linalg.norm(v2) == 0:
            sim = 0
        else:
            sim = (v1.dot(v2))/(numpy.linalg.norm(v1) * numpy.linalg.norm(v2))
        #sim = (self.represent(w1).dot(self.represent(w2)))
        return sim
    
    ## Given a word in the vocabulary, return the vector representatin of the word
    def represent(self, key):
        if key in self.vocab_index:
            return self.vecs[self.vocab_index[key], :]
        else:
            print("Not in Vocabulary: ", key)
            return numpy.zeros(self.dimension)


    #def closest(self, w, n=10):
     #   scores = self.vocab.dot(self.represent(w))
      #  return headpq.nlargest(n, zip(scores, self.vocab_index))

## A class to represent a sequence of word embeddings
class sequential_embedding:
    ## Intialize the class with multiple embeddings
    def __init__(self, year_embeds, **kwargs):
        self.embeds = year_embeds
    
    ## Load embeddings from each year 
    @classmethod
    def load(_class, path, years, **kwargs):
        embeds = collections.OrderedDict()
        for year in years:
            embeds[year] = word_embedding.load_vector(path + "/" + str(year), **kwargs)

        return sequential_embedding(embeds)

    ## Retrieve embeddings from a given year
    def get_embed(self, year):
        return self.embeds[year]
    
    ## Create a time series of similarity scores for two words
    def get_time_sims(self, w1, w2):
        time_sims = collections.OrderedDict()
        for year, embed in self.embeds.items():
            if (w1 in embed.vocab) and (w2 in embed.vocab):
                time_sims[year] = embed.similarity(w1, w2)
            else:
                time_sims[year] = 0
        return time_sims


        