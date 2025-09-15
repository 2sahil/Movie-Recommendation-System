from pathlib import Path
import pandas as pd
import gradio as gr
from src.recommender import HybridRecommender  # adjust if your module name differs

DATA_DIR = Path("data")
RATINGS_CSV = DATA_DIR / "ratings.csv"
MOVIES_CSV  = DATA_DIR / "movies.csv"

recommender = HybridRecommender().fit(str(RATINGS_CSV), str(MOVIES_CSV))

def live_search(q: str):
    """Return up to 10 title suggestions for the searchbox dropdown."""
    if q and len(q.strip()) > 2:
        return recommender.search_titles(q.strip(), top_n=10)
    return []

def make_recs(title: str, top_n: int):
    """Return a DataFrame of recommendations for display."""
    try:
        df = recommender.recommend(title, top_n=top_n)
        return df
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})

with gr.Blocks(title="Movie Recommender") as demo:
    gr.Markdown("# 🎬 Movie Recommender")
    with gr.Row():
        q = gr.Textbox(label="Search titles", placeholder="Type part of a movie title…")
        topn = gr.Slider(1, 20, value=5, step=1, label="Top N")
    matches = gr.Dropdown(label="Matches", choices=[], interactive=True)
    table = gr.Dataframe(label="Recommendations", wrap=True)
    q.change(fn=live_search, inputs=q, outputs=matches)
    # wiring: dropdown selection -> recommendation table
    matches.change(fn=make_recs, inputs=[matches, topn], outputs=table)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
