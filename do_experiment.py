import load_vectors
import random
import numpy


def run_experiment(occupations, start_year, end_year):
    sims = {}
    for i in range(start_year, end_year, 10):
        sims[i] = []
    embeddings = load_vectors.sequential_embedding.load("../sgns", range(start_year, end_year, 10))
    
    for occ in occupations:
        he_time_sims = embeddings.get_time_sims("he", occ)
        she_time_sims = embeddings.get_time_sims("she", occ)
        for year, sim in he_time_sims:
            year_sims = sims[year].append(("he", occ, sim))
            sims[year] = year_sims

        for year, sim in she_time_sims:
            year_sims = sims[year].append("she", occ, sim)
            sims[year] = year_sims
    
    return sims



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


def create_baselines():
    baselines = []
    for year in range(1800, 2000, 10):
        baselines.append(create_baseline("../sgns", year))
    with open("baseline.txt", 'w') as f:
        f.writelines(str(baselines))
    

    

if __name__ == "__main__":
    #embeddings = load_vectors.sequential_embedding.load("../sgns", range(1960, 2000, 10))

     
    #create_baselines()

    extreme_she = ["homemaker", "nurse", "receptionist", "librarian", "socialite", "hairdresser", "nanny", "bookkeeper", "stylist", "housekeeper"]
    extreme_he = ["maestro", "skipper", "protege", "philosopher", "captain", "architect", "financier", "warrior", "broadcaster", "magician"]
    all_occupations = extreme_he + extreme_she 

    print(run_experiment(all_occupations, 1960, 1970)


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
    