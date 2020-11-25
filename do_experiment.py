import load_vectors
import random
import numpy



## Pick some random distribution of words 
## Compute Cosine Similarity of for this random distribution of words over different years

## Questions for Xanda: How to do the best baseline? How to credit a github? Best way to report results? 


## Create Random Baseline
def create_baseline(path, year):
    embedding = load_vectors.word_embedding.load_vector(path + "/" + str(year))
    random_sims = []
    for i in range(0,50000):
        sim = embedding.similarity(random.choice(embedding.vocab), random.choice(embedding.vocab))
        #ensure sims are not exactly the same
        if sim < 1:
            random_sims.append(sim)
    
    baseline = numpy.mean(numpy.array(random_sims))
    return baseline


## Takes Too Long to Run!! 
def all_similarities(vectors):
    result = vectors.dot(vectors.T)
    print(numpy.mean(result))


if __name__ == "__main__":
    embeddings = load_vectors.sequential_embedding.load("../sgns", range(1960, 2000, 10))
    print(create_baseline("../sgns", 1960))
    #print(get_random_words("../sgns", 1970))
    #print(get_random_words("../sgns", 1980))
    #print(get_random_words("../sgns", 1990))

    '''
    time_sims = embeddings.get_time_sims("she", "homemaker")
    for year, sim in time_sims.items():
        print("{}: {}".format(year,sim))
    
    time_sims = embeddings.get_time_sims("he", "homemaker")
    for year, sim in time_sims.items():
        print("{}: {}".format(year,sim))

    time_sims = embeddings.get_time_sims("he", "skipper")
    for year, sim in time_sims.items():
        print("{}: {}".format(year,sim))

    time_sims = embeddings.get_time_sims("she", "skipper")
    for year, sim in time_sims.items():
        print("{}: {}".format(year,sim))
    '''
    