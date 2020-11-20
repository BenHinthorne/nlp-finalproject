import load_vectors

## Pick some random distribution of words 
## Compute Cosine Similarity of for this random distribution of words over different years

def get_random_words(path, year):
    embedding = load_vectors.word_embedding.load_vector(path + "/" + str(year))
    print(embedding.dimension)

if __name__ == "__main__":
    embeddings = load_vectors.sequential_embedding.load("../sgns", range(1980, 2000, 10))
    get_random_words("../sgns", 1980)
    time_sims = embeddings.get_time_sims("she", "homemaker")
    print("Similarity between gay and lesbian from 1980 to 1990:")
    for year, sim in time_sims.items():
        print(sim)


