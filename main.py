import sys

from lsh_netflix import load_data, create_user_movie_matrix
from lsh_netflix import generate_minhash_signatures, lsh_find_similar_pairs
from lsh_netflix import verify_similar_pairs, write_results, plot_similarities

def main():
    if len(sys.argv) > 1:
        random_seed = int(sys.argv[1])
    else:
        random_seed = 42
    
    signature_length = 100
    bands = 10
    rows_per_band = signature_length // bands # floor division
    similarity_threshold = 0.5
    
    print("Loading data...")
    data = load_data()
    
    print("Creating user-movie matrix...")
    user_movie_matrix, user_to_idx = create_user_movie_matrix(data)
    
    print("Generating minhash signatures...")
    signatures = generate_minhash_signatures(user_movie_matrix, signature_length, random_seed)
    
    print("Finding candidate pairs with LSH...")
    candidate_pairs = lsh_find_similar_pairs(signatures, bands, rows_per_band)
    
    print(f"Found {len(candidate_pairs)} candidate pairs")
    print("Verifying similarity...")
    
    similar_pairs, similarities = verify_similar_pairs(
        candidate_pairs, user_movie_matrix, user_to_idx, similarity_threshold)
    
    print(f"Found {len(similar_pairs)} similar pairs with Jaccard similarity > {similarity_threshold}")
    
    write_results(similar_pairs)
    print("Results written to result.txt")

    plot_similarities(similarities, similarity_threshold)

if __name__ == "__main__":

    main()
