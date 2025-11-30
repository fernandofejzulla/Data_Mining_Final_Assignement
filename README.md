## Prerequisites

To run this project, install dependencies with:
```bash
pip install -r requirements.txt
```

## How It Works

The project runs in two main stages.

### Stage 1: Data Preparation and Matrix Construction

The script loads a NumPy .npy dataset of (user, movie, rating) input and builds a sparse matrix where rows represent users and columns represent movies they interacted with.

This stage:
* Loads user_movie_rating.npy
* Extracts unique users and movies
* Builds the binary rating matrix
* Allows optional seed input for reproducibility

This matrix is later used for similarity detection.

### Stage 2: MinHash Signatures and Locality Sensitive Hashing

User rating patterns are compared through MinHash and LSH. MinHash compresses rating vectors into short signatures that approximate Jaccard similarity while LSH groups similar signatures efficiently.

What happens here:
1) MinHash signatures are generated for every user
2) Signatures are split into bands and inserted into LSH buckets
3) Candidate similar users are retrieved and verified by exact Jaccard similarity
4) Similar pairs above threshold are saved into result.txt and visualized in a scatter plot


## How to Run

To run the script:
```bash
python main.py
```

To use your own random_seed when running:
```
python main.py <your_seed>
```

The script loads data, builds signatures, executes LSH, outputs similar user pairs to result.txt and shows a scatter plot of similarity values.