import numpy as np
import pandas as pd
from tqdm import tqdm

from llm_lit.config import HFEmbeddingsSettings
from llm_lit.dim_reduction import run_umap

np.random.seed(42)

embedding_model = HFEmbeddingsSettings().build_model()
json_file = "/lambda_stor/homes/heng.ma/dataset/arxiv-metadata-oai-snapshot.json"

weight_list = [
    "physics",
    "High Energy Physics",
    "astrophysics",
    "mathematics",
    "Computer Science",
]
weight_embeddings = [embedding_model.embed_query(i) for i in weight_list]

df = pd.read_json(json_file, lines=True)

df = df.iloc[::100]
emb_df = []
for i, row in tqdm(df.iterrows(), total=len(df), desc="Embedding Prog:"):
    emb = embedding_model.embed_query(row.abstract)
    nat_dict = row.to_dict()
    nat_dict["embedding"] = emb
    for name, w_emb in zip(weight_list, weight_embeddings):
        name = name.replace(" ", "")
        nat_dict[f"embedding_{name}"] = np.sqrt(np.abs(w_emb)) * emb
    emb_df.append(nat_dict)

emb_df = pd.DataFrame(emb_df)

for name in weight_list:
    name = name.replace(" ", "")
    umap_emb = run_umap(
        emb_df[f"embedding_{name}"].to_list(), n_components=2, metric="cosine"
    )
    emb_df[f"umap_{name}_0"] = umap_emb[:, 0]
    emb_df[f"umap_{name}_1"] = umap_emb[:, 1]

emb_df.to_pickle("arxiv_hf_weighted.pkl")

# fig = px.scatter(
#     df.iloc[::],
#     x="umap_0",
#     y="umap_1",
#     color="cat",
#     hover_data=["id", "title", "categories"],
#     render_mode="webgl",
# )
# fig.update_traces(marker=dict(size=4))
# fig.write_html("umap_hf.html")
