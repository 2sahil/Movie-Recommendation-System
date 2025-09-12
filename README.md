# 🎥 Movie Recommendation System (MovieLens)

A hybrid movie recommender system using **Truncated SVD** + **genre features** + **cosine kNN** on the [MovieLens 25M dataset](https://grouplens.org/datasets/movielens/25m/).  
Includes a **search widget** and a clean **modular Python implementation**.

---

## ✨ Features

- Filters out **inactive users/movies** (less than 5 user ratings, less than 20 movie ratings)
-  Learns **movie embeddings** with TruncatedSVD
-  Adds **genre features** with configurable weight
-  Finds **similar movies** using cosine similarity (k-NN)
-  Includes **TF-IDF title search** + fuzzy matching

---

## 📂 Repository Structure

