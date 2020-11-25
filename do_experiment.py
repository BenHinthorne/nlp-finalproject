import load_vectors
import random
import numpy
import pandas as pd
import matplotlib.pyplot as plt


def run_experiment(occupations, start_year, end_year):
    sims = {}
    for i in range(start_year, end_year, 10):
        sims[i] = []
    embeddings = load_vectors.sequential_embedding.load("../sgns", range(start_year, end_year, 10))

    she_scores = {}
    he_scores = {}
    print(sims)
    for occ in occupations:
        he_time_sims = embeddings.get_time_sims("he", occ)
        she_time_sims = embeddings.get_time_sims("she", occ)
        he_occ_scores = []
        she_occ_scores = []
        for year, sim in he_time_sims.items():
            he_occ_scores.append(sim)

        for year, sim in she_time_sims.items():
            she_occ_scores.append(sim)

        he_scores[occ] = he_occ_scores
        she_scores[occ] = she_occ_scores

    return he_scores, she_scores



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
        for val in baselines:
            f.write(str(val))
            f.write("\n")

def read_baselines(start_year, end_year):
    with open("baseline.txt", 'r') as f:
        baselines = [line.strip() for line in f]
    year = 1800
    base_dict = {}
    for val in baselines:
        base_dict[year] = val
        year += 10
    
    needed_vals = []
    for i in range(start_year, end_year, 10):
        needed_vals.append(base_dict[i])
    return needed_vals

if __name__ == "__main__":
    #embeddings = load_vectors.sequential_embedding.load("../sgns", range(1960, 2000, 10))

     
    #create_baselines()

    extreme_she = ["homemaker", "nurse", "receptionist", "librarian", "socialite", "hairdresser", "nanny", "bookkeeper", "stylist", "housekeeper"]
    extreme_he = ["maestro", "skipper", "protege", "philosopher", "captain", "architect", "financier", "warrior", "broadcaster", "magician"]
    all_occupations = extreme_he + extreme_she 

    start_year = 1900
    end_year = 2000
    cols = ["year"]
    data = {}
    years = []
    he_scores, she_scores = run_experiment(all_occupations, start_year, end_year)
    for i in range(start_year, end_year, 10):
        years.append(i)
    data["year"] = years
    df = pd.DataFrame(data)

    baselines = read_baselines(start_year, end_year)
    df["baseline"] = baselines
    for key in he_scores:
        he_key = "he/" + key
        she_key = "she/" + key
        df[he_key] =  he_scores[key]
        df[she_key] = she_scores[key]
  
    print(df)

    #ax = plt.gca()
    #df.plot(kind='scatter', x='year', y='baseline', ax=ax)
    #df.plot(kind='scatter', x='year', y='he/housekeeper', ax=ax)
    #df.plot(kind='scatter', x='year', y='she/housekeeper', ax=ax)
    #plt.savefig("difplot.png")
    df[["year", "baseline", "he/housekeeper", "she/housekeeper"]].plot(x='year', y=['baseline'] kind="scatter")
    #df.plot(x='year', y=["baseline", "he/housekeeper", "she/housekeeper"], kind="scatter")
    plt.savefig('plot.png')
    #df.plot(x='year', y=["baseline", "he/housekeeper"], kind="scatter")
    #plt.savefig('plot2.png')



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