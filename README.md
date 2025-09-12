#  Movie Recommendation System (MovieLens)

A hybrid movie recommender system using **Truncated SVD** + **genre features** + **cosine kNN** on the [MovieLens 25M dataset](https://grouplens.org/datasets/movielens/25m/).  
Includes a **search widget** and a clean **modular Python implementation**.

---

##  Features

- Filters out **inactive users/movies** (less than 5 user ratings, less than 20 movie ratings)
-  Learns **movie embeddings** with TruncatedSVD
-  Adds **genre features** with configurable weight
-  Finds **similar movies** using cosine similarity (k-NN)
-  Includes **TF-IDF title search** + fuzzy matching

---

## 📥 Setup & Installation

Run these commands step by step in your terminal (or command prompt):

```bash
git clone https://github.com/<your-username>/movie-recommender.git
cd movie-recommender
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

Download ratings.csv and movies.csv from MovieLens 25M
and place them into a data/ folder:

movie-recommender/
└── data/
    ├── ratings.csv
    └── movies.csv
Run this command to get recommendations for a movie:
python scripts/run_demo.py --ratings data/ratings.csv --movies data/movies.csv --title "The Godfather (1972)"

Jupyter Notebook Demo:
jupyter notebook notebooks/demo.ipynb

License & Data

This repository uses the publicly available MovieLens 25M dataset but does not redistribute it (see DATASET.md).
Please download from the official MovieLens website
.
