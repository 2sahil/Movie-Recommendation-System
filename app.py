from pathlib import Path
import re
import numpy as np
import pandas as pd
import gradio as gr
from sklearn.metrics.pairwise import cosine_similarity

from src.recommender import HybridRecommender 
DATA_DIR = Path("data")
RATINGS = DATA_DIR / "ratings.csv"
MOVIES  = DATA_DIR / "movies.csv"

rec = HybridRecommender().fit(str(RATINGS), str(MOVIES))

_movies = rec.movies                   
_vec     = rec.vectorizer               
_tfidf   = rec.tfidf                     

_kept_ids = set(int(x) for x in rec.unique_movies)
_kept_mask = _movies["movieId"].isin(list(_kept_ids)).values
_kept_idxs = np.where(_kept_mask)[0]

def _clean(t: str) -> str:
    return re.sub(r"[^a-zA-Z0-9 ]", "", t).lower()

def live_search(query: str):
    """Return up to 10 recommendable titles similar to the query."""
    q = (query or "").strip()
    if len(q) < 3 or _vec is None or _tfidf is None or _movies is None or _kept_idxs.size == 0:
        return gr.update(choices=[], value=None)

    q_vec = _vec.transform([_clean(q)])
    sim = cosine_similarity(q_vec, _tfidf).flatten()

    pool = min(50, _kept_idxs.size)
    top = _kept_idxs[np.argpartition(sim[_kept_idxs], -pool)[-pool:]]
    top = top[np.argsort(sim[top])[::-1]]

    seen, choices = set(), []
    for i in top:
        title = str(_movies.iloc[i]["title"])
        if title not in seen:
            seen.add(title)
            choices.append(title)
        if len(choices) >= 10:
            break

    return gr.update(choices=choices, value=(choices[0] if choices else None))

def make_recs(title: str, top_n: int):
    """Return recommendations for the selected title (or a friendly hint)."""
    if not title:
        return pd.DataFrame({"message": ["Pick a title from the list"]})
    try:
        return rec.recommend(title, top_n=int(top_n))
    except Exception as e:
        return pd.DataFrame({"error": [str(e)], "hint": ["Try another title or type ≥3 chars"]})

with gr.Blocks(title=" Movie Recommender") as demo:
    gr.Markdown(" # Movie Recommender")
    gr.Markdown("Start typing (≥3 characters). Pick a title from the list to see similar movies.")

    with gr.Row():
        q = gr.Textbox(label="Search titles", placeholder="e.g. Amer, Toy, Matrix…", autofocus=True)
        topn = gr.Slider(1, 20, value=5, step=1, label="Top N")

    matches = gr.Radio(label="Matches", choices=[], interactive=True)

    table = gr.Dataframe(label="Recommendations")

    q.change(fn=live_search, inputs=q, outputs=matches)
    matches.change(fn=make_recs, inputs=[matches, topn], outputs=table)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, share=False)
