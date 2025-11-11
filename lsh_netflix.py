import random
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from collections import defaultdict

def load_data():
    # load data
    data = np.load('data/user_movie_rating.npy')
    return data

def create_user_movie_matrix(data):
    # get unique users and movies
    users = np.unique(data[:, 0])
    movies = np.unique(data[:, 1])
    
    # create mapping from user/movie id to index
    user_to_idx = {user_id: idx for idx, user_id in enumerate(users)}
    movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(movies)}
    
    # create sparse binary matrix (1 if user rated movie, 0 otherwise)
    rows = [user_to_idx[user] for user in data[:, 0]]
    cols = [movie_to_idx[movie] for movie in data[:, 1]]
    
    # binary matrix: 1 if rated, 0 otherwise (ratings don't matter)
    user_movie_matrix = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(users), len(movies)))  
    
    return user_movie_matrix, user_to_idx

def generate_minhash_signatures(user_movie_matrix, signature_length, random_seed):
    n_users, n_movies = user_movie_matrix.shape
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # generate random permutations using hash functions
    max_hash = (1 << 31) - 1
    a = np.random.randint(1, max_hash, size=signature_length, dtype=np.int64)
    b = np.random.randint(0, max_hash, size=signature_length, dtype=np.int64)
    movie_indices = np.arange(n_movies, dtype=np.int64)
    all_hashes = (a[:, None] * movie_indices[None, :] + b[:, None]) % max_hash

    signatures = np.full((signature_length, n_users), np.inf, dtype=np.float64)
    
    # for each movie (column), update signatures
    indptr = user_movie_matrix.indptr
    indices = user_movie_matrix.indices

    for user_idx in range(n_users):
        start_ptr = indptr[user_idx]
        end_ptr = indptr[user_idx + 1]
        user_movies = indices[start_ptr:end_ptr]
        if user_movies.size == 0:
            continue
        user_hashes = all_hashes[:, user_movies]
        signatures[:, user_idx] = user_hashes.min(axis=1)
    
    return signatures

def lsh_find_similar_pairs(signatures, bands, rows_per_band, threshold=0.5):
    n_users = signatures.shape[1]
    candidate_pairs = set()
    
    # process each band
    for band_idx in range(bands):
        start_row = band_idx * rows_per_band
        end_row = min((band_idx + 1) * rows_per_band, signatures.shape[0])
        
        # create buckets for this band
        buckets = defaultdict(list)
        
        # hash each user's signature segment to bucket
        for user_idx in range(n_users):
            band_signature = tuple(signatures[start_row:end_row, user_idx])
            bucket_id = hash(band_signature)
            buckets[bucket_id].append(user_idx)
        
        # find candidate pairs within each bucket
        for bucket_users in buckets.values():
            if len(bucket_users) > 1:
                # add all pairs from this bucket as candidates
                for i in range(len(bucket_users)):
                    for j in range(i + 1, len(bucket_users)):
                        u1, u2 = bucket_users[i], bucket_users[j]
                        if u1 < u2:
                            candidate_pairs.add((u1, u2))
    
    return candidate_pairs

def verify_similar_pairs(candidate_pairs, user_movie_matrix, user_to_idx, threshold=0.5):
    idx_to_user = {idx: user for user, idx in user_to_idx.items()}
    similar_pairs = []
    similarities = []
    
    for u1_idx, u2_idx in candidate_pairs:
        # calculate Jaccard similarity
        set1 = set(user_movie_matrix[u1_idx].nonzero()[1])
        set2 = set(user_movie_matrix[u2_idx].nonzero()[1])
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        similarity = intersection / union if union > 0 else 0
        
        if similarity > threshold:
            user1_id = idx_to_user[u1_idx]
            user2_id = idx_to_user[u2_idx]
            if user1_id < user2_id:
                similar_pairs.append((user1_id, user2_id))
            else:
                similar_pairs.append((user2_id, user1_id))
            similarities.append(similarity)
    
    return similar_pairs, similarities

def write_results(similar_pairs, filename="result.txt"):
    with open(filename, 'w') as f:
        for user1, user2 in similar_pairs:
            f.write(f"{user1},{user2}\n")

def plot_similarities(similarities, threshold=0.5):
    # plot of Jaccard similarities (sorted)
    if not similarities:
        print("No similar pairs to plot.")
        return
    
    sims_sorted = sorted(similarities)
    x = np.arange(1, len(sims_sorted) + 1)

    plt.figure()
    plt.scatter(x, sims_sorted, s=5)
    plt.xlabel("Most similar pairs")
    plt.ylabel("Jaccard Similarity")
    plt.title(f"{len(sims_sorted)} pairs with JS > {threshold}")
    plt.tight_layout()
    plt.show()
