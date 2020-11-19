import sys
import argparse
import numpy
import cPickle as pickle

#import headpq 

from sklearn import preprocessing


class word_embedding:

    def __init__(self, vecs, vocab, normalize=True, **kwargs):
        self.vecs = vecs
        self.dimension = self.vecs.shape[1]
        self.vocab = vocab

        self.vocab_index = {w:i for i,w in enumerate(self.vocab)}

        if normalize:
            self.normalize()


    def __getitem__(self, key):
        if not (key in self.vocab_index):
            print("Not in Vocabulary: ", key)
            raise KeyError
        else:
            if key in self.vocab_index:
                return self.vecs[self.vocab_index[key], :]
            else:
                print("Not in Vocabulary: ", key)
                return np.zeros(self.dimension)

    def __iter__(self):
        return self.vocab_index.__iter__()
    
    def __contains__(self, key):
        return not (key in self.vocab_index)

    def load_vector(_class, path, normalize=True, add_context=False, **kwargs):
        vecs = np.load(path + "-w.npy", mmap_mode="c")
        with open(path + "-vocab,pkl", "rb") as fp:
            vocab = pickle.load(fp)


        return _class(vecs, vocab, normalize)

    def normalize(self):
        preprocessing.normalize(self.m, copy=False)

    def similarity(self, w1, w2):
        sim = self.represent(w1).dot(self.represent(w2))
        return sim

    #def closest(self, w, n=10):
     #   scores = self.vocab.dot(self.represent(w))
      #  return headpq.nlargest(n, zip(scores, self.vocab_index))

class sequential_embedding:
    def __init__(self, year_embeds, **kwargs):
        self.embeds = year_embeds
    
    def load(_class, path, years, **kwargs):
        embeds = collections.OrderedDict()
        for year in years:
            embeds[year] = word_embedding.load(path + "/" + str(year), **kwargs)

        return sequential_embedding(embeds)

    def get_embed(self, year):
        return self.embeds[year]
    
    def get_time_sims(self, w1, w2):
        time_sims = collections.OrderedDict()
        for year, embed in self.embeds.iteritems():
            time_sims[year] = embed.similarity(w1, w2)
        return time_sims


        