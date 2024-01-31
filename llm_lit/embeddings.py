from argparse import ArgumentParser

from llm_lit.config import EmbeddingSettingsTypes
from llm_lit.utils import BaseModel


class llm_emb_cfg(BaseModel):
    embedding_setup: EmbeddingSettingsTypes


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()
    cfg = llm_emb_cfg.from_yaml(args.config)
