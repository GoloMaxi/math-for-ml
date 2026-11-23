# Sprint 01: Finding Similar Words

The goal of this sprint is to find similar words using GloVe embeddings.
Each word is represented by a 100-dimensional vector.
I compare cosine similarity and Euclidean distance to see how normalization
and the choice of metric affect which words are returned as nearest neighbors.

For 1_000 x 1_000 pairs of 100-dimensional vectors, the vectorized implementftion took 0.0213 s,
compared with 2.9216 s for the nested loops (136.9× speedup)