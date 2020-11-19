from load_vectors import SequentialEmbedding

if __name__ == "__main__":
    embeddings = SequentialEmbedding.load("../sgns", range(1980, 2000, 10))
    time_sims = embeddings.get_time_sims("lesibian", "gay"):
    print("Similarity between gay and lesbian from 1980 to 1990:")
    for year, sim in time_sims.iteritems():
        print(year + sim)
