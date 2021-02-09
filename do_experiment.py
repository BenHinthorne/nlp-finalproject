import load_vectors
import random
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import scipy


def load_vector(year):
    embeddings = load_vectors.word_embedding.load_vector("../sgns/1990")
    return embeddings 

def cos_sim(embeddings, w1, w2):
    return embeddings.similarity(w1,w2)

def create_single_baseline(embedding):
    random_sims = []
    for i in range(0,50000):
        sim = cos_sim(embedding, random.choice(embedding.vocab), random.choice(embedding.vocab))
        if sim < 1:
            random_sims.append(sim)
    
    baseline = numpy.mean(numpy.array(random_sims))
    return baseline

def run_small_test():
    embeds = load_vector(1990)
    same_sim = cos_sim(embeds, "world", "world")
    print("Cosine similarity of the same word: {}".format(same_sim))
    print("Baseline random similarity: {}".format(create_single_baseline(embeds)))

## Given a list of occupations, returns the similarity scores of the occupations with 
## pronouns as a time series from the specified start year to the specified end year
def run_experiment(occupations, start_year, end_year):

    ## Initialize Sequential Embeddings
    sims = {}
    for i in range(start_year, end_year, 10):
        sims[i] = []
    embeddings = load_vectors.sequential_embedding.load("../sgns", range(start_year, end_year, 10))

    she_scores = {}
    he_scores = {}

    ## For Each Occupation, get the time similarities with pronouns
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



## Creates a random baseline of similarity by choosing 500000 random pairs of words to calculate
## the similarity score, and then takes the average
def create_baseline(path, year):
    embedding = load_vectors.word_embedding.load_vector(path + "/" + str(year))
    random_sims = []
    for i in range(0,50000):
        sim = embedding.similarity(random.choice(embedding.vocab), random.choice(embedding.vocab))
        #ensure words are not exactly the same
        if sim < 1:
            random_sims.append(sim)
    
    baseline = numpy.mean(numpy.array(random_sims))
    return baseline

## Creates the baselines and writes them to a file over 1800 to 2000
def create_baselines():
    baselines = []
    for year in range(1800, 2000, 10):
        baselines.append(create_baseline("../sgns", year))
    with open("baseline.txt", 'w') as f:
        for val in baselines:
            f.write(str(val))
            f.write("\n")

## Reads the baselines from a file
def read_baselines(start_year, end_year):
    with open("baseline.txt", 'r') as f:
        baselines = [float(line.strip()) for line in f]
    year = 1800
    base_dict = {}
    for val in baselines:
        base_dict[year] = val
        year += 10
    
    needed_vals = []
    for i in range(start_year, end_year, 10):
        needed_vals.append(base_dict[i])
    return needed_vals

## Creates a plot of cosine similarity for the specified occupation and passed in data frame
def plot_analysis(df, occupation):
    he_label = "he/" + occupation
    she_label = "she/" + occupation
    
    ax1 = df.plot(x='year', y='baseline', color="blue", label="baseline", kind="scatter")
    df.plot(x='year', y=he_label, kind='scatter', color="orange", label=he_label, ax=ax1)
    df.plot(x='year', y=she_label, kind='scatter', color="green", label=she_label, ax=ax1)
    plt.xlabel("Year")
    plt.ylabel("Cosine Similarity") 
    filename = occupation + "_scatter.png"
    plt.savefig(filename)

    df.plot(x='year', y=["baseline", he_label, she_label], kind="line")
    plt.xlabel("Year")
    plt.ylabel("Cosine Similarity") 
    plt.title(occ)
    filename = occupation + "_line.png"
    plt.savefig(filename)

## Plots the difference of cosine similarites at each point in time of he and she (plotting bias score)
def plot_difference(df, occ):
    copy_df = df
    label_1 = "he/" + occ
    label_2 = "she/" + occ
    df['diff'] = abs(copy_df[label_1] - copy_df[label_2])
    z = numpy.polyfit(x=copy_df.loc[:,'year'], y=copy_df.loc[:,'diff'], deg=1)
    p = numpy.poly1d(z)
    df['trendline'] = p(copy_df.loc[:,'year'])

    ax = copy_df.plot(x='year', y='diff', kind="scatter")
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(copy_df.loc[:,'year'], copy_df.loc[:,'diff'])
    r_2 = abs(r_value) ** abs(r_value)
    copy_df.set_index('year', inplace=True)
    slope_label = 'Slope: ' + str(round(slope,7))
    r_label = 'R^2: ' + str(round(r_2,7))

    ax.text(0.85,0.9, slope_label, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    ax.text(0.85,0.8, r_label, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    copy_df.trendline.sort_index(ascending=False).plot(ax=ax)
    plt.xlabel("year")
    plt.ylabel("Bias Score")
    plt.title(occ)
    filename = occ + "diff.png"
    plt.savefig(filename)
    return slope, r_2


if __name__ == "__main__":
    #embeddings = load_vectors.sequential_embedding.load("../sgns", range(1960, 2000, 10))
    #create_baselines()

    run_small_test()

    '''
    ## Define he and she occupations to use
    extreme_she = ["homemaker", "nurse", "receptionist", "librarian", "socialite", "hairdresser", "nanny", "bookkeeper", "stylist", "housekeeper"]
    extreme_he = ["maestro", "skipper", "protege", "philosopher", "captain", "architect", "financier", "warrior", "broadcaster", "magician"]
    all_occupations = extreme_he + extreme_she 

    ## Year boundaries for experiment 
    start_year = 1900
    end_year = 2000

    ## Run the experiment and store in a dataframe 
    cols = ["year"]
    data = {}
    years = []
    he_scores, she_scores = run_experiment(all_occupations, start_year, end_year)
    for i in range(start_year, end_year, 10):
        years.append(i)
    data["year"] = years
    df = pd.DataFrame(data)

    ## Plot the results 
    baselines = read_baselines(start_year, end_year)
    df["baseline"] = baselines
    for key in he_scores:
        he_key = "he/" + key
        she_key = "she/" + key
        df[he_key] =  he_scores[key]
        df[she_key] = she_scores[key]

    print(df)
    df.to_csv("results.csv", index=False)



    with open("differences.txt", 'w') as f:
        for occ in all_occupations:
            copy_df = df.copy()
            new_df = df.copy()
            plot_analysis(copy_df, occ)
            slope, r_2 = plot_difference(new_df, occ)
            line = occ + ", " + str(slope) + ", " + str(r_2)
            f.write(line)
            f.write("\n")





    '''
    ## Saving in case I want to reference later
    '''
    #ax = plt.gca()
    #df.plot(kind='scatter', x='year', y='baseline', ax=ax)
    #df.plot(kind='scatter', x='year', y='he/housekeeper', ax=ax)
    #df.plot(kind='scatter', x='year', y='she/housekeeper', ax=ax)
    #plt.savefig("difplot.png")
    #ax1 = df.plot(x='year', y='baseline', color="green", label="baseline", kind="scatter")
    #df.plot(x='year', y='he/homemaker', kind='scatter', color="red", label="he/homemaker", ax=ax1)
    #df.plot(x='year', y='she/homemaker', kind='scatter', color="blue", label="she/homemaker", ax=ax1)
    #plt.xlabel("Year")
    #plt.ylabel("Cosine Similarity") 
    #df.plot(x='year', y=["baseline", "he/housekeeper", "she/housekeeper"], kind="scatter")
    #plt.savefig('plot.png')
    #df.plot(x='year', y=["baseline", "he/housekeeper"], kind="scatter")
    #plt.savefig('plot2.png')



    #print(get_random_words("../sgns", 1970))
    #print(get_random_words("../sgns", 1980))
    #print(get_random_words("../sgns", 1990))

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
