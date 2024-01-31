import numpy as np
import pandas as pd
import plotly.express as px
from tqdm import tqdm

from llm_lit.config import HFEmbeddingsSettings
from llm_lit.dim_reduction import run_umap

np.random.seed(42)

embedding_model = HFEmbeddingsSettings().build_model()

items = [
    "London",
    "UK",
    "Paris",
    "Bordeaux",
    "Roma",
    "Italia",
    "Berlin",
    "München",
    "Europe",
    "New York",
    "USA",
    "America",
    "Beijing",
    "Asia",
    "Constantinople",
    "City",
    "German city",
    "French city",
    "American city",
    "Wine",
    "Is Paris a French city? ",
    # "What are we hoping for? ",
]
emb_df = []
for i, item in tqdm(enumerate(items), total=len(items), desc="Embedding Prog:"):
    emb = embedding_model.embed_query(item)
    nat_dict = {"name": item}
    nat_dict["embedding"] = emb
    emb_df.append(nat_dict)

emb_df = pd.DataFrame(emb_df)

umap_emb = run_umap(emb_df.embedding.to_list())

emb_df["umap_0"] = umap_emb[:, 0]
emb_df["umap_1"] = umap_emb[:, 1]

emb_df.to_pickle("loc_hf.pkl")


fig = px.scatter(
    emb_df,
    x="umap_0",
    y="umap_1",
    color="name",
    hover_data=["name"],
    render_mode="webgl",
)
fig.update_traces(marker=dict(size=10))
fig.write_html("umap_loc.html")
