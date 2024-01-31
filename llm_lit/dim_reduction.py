import umap
from sklearn.manifold import TSNE


def run_umap(embeddings, **kwargs):
    umap_emb = umap.UMAP(**kwargs).fit_transform(embeddings)
    return umap_emb


def run_tsne(embeddings, **kwargs):
    X_embedded = TSNE(**kwargs).fit_transform(embeddings)
    return X_embedded
