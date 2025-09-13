from __future__ import annotations

import re
import difflib
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HybridRecommender:
    """
    Hybrid movie recommender:
      - Learns latent movie factors via TruncatedSVD on a movie×user rating matrix
      - Adds genre one-hot features (weighted)
      - Uses cosine KNN on the combined feature space for similar-movie lookup
      - Provides TF-IDF + fuzzy title search helper
    """

    def __init__(
        self,
        min_movie_ratings: int = 20,
        min_user_ratings: int = 5,
        n_components: int = 32,
        genre_weight: float = 0.3,
        n_neighbors: int = 6,
        random_state: int = 22,
    ):
        self.min_movie_ratings = min_movie_ratings
        self.min_user_ratings = min_user_ratings
        self.n_components = n_components
        self.genre_weight = genre_weight
        self.n_neighbors = n_neighbors
        self.random_state = random_state

        # Fitted artifacts
        self.movies: pd.DataFrame | None = None
        self.unique_movies: np.ndarray | None = None
        self.unique_users: np.ndarray | None = None
        self.movie_to_idx: dict[int, int] | None = None
        self.user_to_idx: dict[int, int] | None = None
        self.hybrid_feats: csr_matrix | None = None
        self.knn: NearestNeighbors | None = None

        # Search artifacts
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf = None
        self.clean_choices: list[str] | None = None
        self.clean_to_title: dict[str, str] | None = None

    # ---------- Public API ----------

    def fit(self, ratings_csv: str, movies_csv: str) -> "HybridRecommender":
        """
        Build the hybrid feature matrix and fit KNN.
        Expects MovieLens-style ratings.csv and movies.csv.
        """
        ratings = pd.read_csv(ratings_csv)
        movies = pd.read_csv(movies_csv)
        self.movies = movies.copy()

        # --- Filter activity: movies with >= min_movie_ratings; users with >= min_user_ratings
        movie_counts = ratings["movieId"].value_counts()
        popular_movies = movie_counts[movie_counts >= self.min_movie_ratings].index
        ratings = ratings[ratings["movieId"].isin(popular_movies)]

        user_counts = ratings["userId"].value_counts()
        active_users = user_counts[user_counts >= self.min_user_ratings].index
        ratings = ratings[ratings["userId"].isin(active_users)]

        # --- Index mapping for sparse matrix
        unique_movies = ratings["movieId"].unique()
        unique_users = ratings["userId"].unique()
        self.unique_movies = unique_movies
        self.unique_users = unique_users

        self.movie_to_idx = {mid: i for i, mid in enumerate(unique_movies)}
        self.user_to_idx = {uid: i for i, uid in enumerate(unique_users)}

        rows = ratings["movieId"].map(self.movie_to_idx)
        cols = ratings["userId"].map(self.user_to_idx)
        data = ratings["rating"].astype(np.float32)

        # movie×user sparse rating matrix (CSR)
        sparse_ui = coo_matrix(
            (data, (rows, cols)),
            shape=(len(unique_movies), len(unique_users))
        ).tocsr()

        # --- Latent movie embeddings via TruncatedSVD
        svd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        movie_embeds = svd.fit_transform(sparse_ui).astype(np.float32)
        svd_sparse = csr_matrix(movie_embeds)

        # --- Genre one-hot features (weighted)
        genre_dummies = (
            movies.set_index("movieId")["genres"]
            .str.get_dummies(sep="|")
            .reindex(index=unique_movies, fill_value=0)
        )
        genre_sparse = csr_matrix(genre_dummies.values * float(self.genre_weight))

        # --- Hybrid feature space: [latent | genres]
        self.hybrid_feats = hstack([svd_sparse, genre_sparse], format="csr")

        # --- Cosine KNN over hybrid features
        self.knn = NearestNeighbors(
            n_neighbors=int(self.n_neighbors),
            metric="cosine",
            algorithm="brute",
        )
        self.knn.fit(self.hybrid_feats)

        # --- Build title search index (over all movies)
        self.movies["clean_title"] = self.movies["title"].apply(_clean_title)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.tfidf = self.vectorizer.fit_transform(self.movies["clean_title"])
        self.clean_choices = self.movies["clean_title"].tolist()
        self.clean_to_title = dict(zip(self.clean_choices, self.movies["title"]))

        return self

    def recommend(self, movie_title: str, top_n: int = 5) -> pd.DataFrame:
        """
        Recommend top-N similar movies to the first title match (case-insensitive).
        """
        _ensure_fitted(self)
        assert self.movies is not None

        matches = self.movies[self.movies["title"].str.contains(movie_title, case=False, regex=False)]
        if matches.empty:
            raise ValueError(f"No movies found matching '{movie_title}'")

        movie_id = int(matches.iloc[0]["movieId"])
        rec_ids, rec_scores = self._recommend_by_id(movie_id, top_n=top_n)
        titles = self.movies.set_index("movieId").loc[rec_ids]["title"].values
        return pd.DataFrame({"movieId": rec_ids, "title": titles, "score": rec_scores})

    def search_titles(self, query: str, top_n: int = 10, fuzzy_cutoff: float = 0.5) -> list[str]:
        """
        TF-IDF + fuzzy search over movie titles. Returns up to top_n unique titles.
        """
        _ensure_fitted(self)
        assert self.vectorizer is not None and self.tfidf is not None
        assert self.clean_choices is not None and self.clean_to_title is not None

        q_clean = _clean_title(query)
        q_vec = self.vectorizer.transform([q_clean])
        sim = cosine_similarity(q_vec, self.tfidf).flatten()

        idxs = np.argpartition(sim, -top_n)[-top_n:]
        idxs = idxs[np.argsort(sim[idxs])[::-1]]
        tfidf_titles = self.movies.iloc[idxs]["title"].tolist()  # type: ignore

        fuzzy_clean = difflib.get_close_matches(q_clean, self.clean_choices, n=top_n, cutoff=fuzzy_cutoff)
        fuzzy_titles = [self.clean_to_title[c] for c in fuzzy_clean]

        combined: list[str] = []
        for t in tfidf_titles + fuzzy_titles:
            if t not in combined:
                combined.append(t)
            if len(combined) >= top_n:
                break
        return combined

    # ---------- Internal helpers ----------

    def _recommend_by_id(self, movie_id: int, top_n: int = 5):
        _ensure_fitted(self)
        if movie_id not in self.movie_to_idx:  # type: ignore
            raise ValueError(f"Movie ID {movie_id} was filtered out (too few ratings).")
        idx = self.movie_to_idx[movie_id]  # type: ignore
        dists, idxs = self.knn.kneighbors(self.hybrid_feats[idx], n_neighbors=top_n + 1)  # type: ignore
        rec_idxs = idxs.flatten()[1:]
        rec_ids = [int(self.unique_movies[i]) for i in rec_idxs]  # type: ignore
        scores = 1.0 - dists.flatten()[1:]
        return rec_ids, scores


# ---------- Module-level utilities ----------

def _clean_title(title: str) -> str:
    return re.sub(r"[^a-zA-Z0-9 ]", "", title).lower()


def _ensure_fitted(obj: HybridRecommender):
    if any(
        v is None
        for v in [
            obj.movies,
            obj.unique_movies,
            obj.movie_to_idx,
            obj.hybrid_feats,
            obj.knn,
            obj.vectorizer,
            obj.tfidf,
        ]
    ):
        raise RuntimeError("Recommender is not fitted. Call fit(ratings_csv, movies_csv) first.")
