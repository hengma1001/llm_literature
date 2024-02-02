import pandas as pd
import plotly.express as px
from tqdm import tqdm

from llm_lit.config import LlamaCppSettings
from llm_lit.dim_reduction import run_umap

n_gpu_layers = 40
embedding_model = LlamaCppSettings(
    model_path="/homes/heng.ma/Research/md_pkgs/llm-scripts/langchain/llama/llama-2-7b.Q4_K_M.gguf",
    # n_gpu_layers=n_gpu_layers,
).build_model()

json_file = "/lambda_stor/homes/heng.ma/dataset/arxiv-metadata-oai-snapshot.json"
df = pd.read_json(json_file, lines=True)

df = df.iloc[::100]
emb_df = []
for i, row in tqdm(df.iterrows(), total=len(df), desc="Embedding Prog:"):
    emb = embedding_model.embed_query(row.abstract)
    nat_dict = row.to_dict()
    nat_dict["embedding"] = emb
    emb_df.append(nat_dict)

emb_df = pd.DataFrame(emb_df)


umap_emb = run_umap(emb_df.embedding.to_list())

emb_df["umap_0"] = umap_emb[:, 0]
emb_df["umap_1"] = umap_emb[:, 1]

emb_df.to_pickle("arxiv_hf.pkl")


fig = px.scatter(
    df.iloc[::],
    x="umap_0",
    y="umap_1",
    color="cat",
    hover_data=["id", "title", "categories"],
    render_mode="webgl",
)
fig.update_traces(marker=dict(size=4))
fig.write_html("umap_llama.html")
