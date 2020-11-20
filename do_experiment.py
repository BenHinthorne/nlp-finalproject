import load_vectors

## Pick some random distribution of words 
## Compute Cosine Similarity of for this random distribution of words over different years


if __name__ == "__main__":
    embeddings = load_vectors.sequential_embedding.load("../sgns", range(1980, 2000, 10))
    for year, embed in embeddings.items():
        print(embed.vocab[0:10])
    time_sims = embeddings.get_time_sims("she", "homemaker")
    print("Similarity between gay and lesbian from 1980 to 1990:")
    for year, sim in time_sims.items():
        print(sim)
