from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional, Union

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import LlamaCppEmbeddings
from pydantic import Field

from llm_lit.utils import BaseModel


class BaseEmbeddingSettings(BaseModel, ABC):
    """Encoder settings (model, model_kwargs)"""

    name: Literal[""] = ""

    @abstractmethod
    def build_model(self) -> Union[LlamaCppEmbeddings, HuggingFaceEmbeddings]:
        """Create a new Parsl configuration.
        Parameters
        ----------

        Returns
        -------
        Config
            Parsl configuration.
        """
        ...


class HFEmbeddingsSettings(BaseEmbeddingSettings):
    """Pull model from HuggingFace with model card"""

    name: Literal["HFEmbeddings"] = "HFEmbeddings"
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    """Model name to use."""
    cache_folder: Optional[str] = None
    """Path to store models. Can be also set by SENTENCE_TRANSFORMERS_HOME environment variable."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass to the model."""
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the `encode` method of the model."""
    multi_process: bool = False
    """Run encode() on multiple GPUs."""

    def build_model(self) -> HuggingFaceEmbeddings:
        model_setup = self.dict().copy()
        model_setup.pop("name")
        return HuggingFaceEmbeddings(**model_setup)


class LlamaCppSettings(BaseEmbeddingSettings):
    name: Literal["LlamaCpp"] = "LlamaCpp"
    model_path: str

    n_ctx: int = Field(512, alias="n_ctx")
    """Token context window."""

    n_parts: int = Field(-1, alias="n_parts")
    """Number of parts to split the model into. 
    If -1, the number of parts is automatically determined."""

    seed: int = Field(-1, alias="seed")
    """Seed. If -1, a random seed is used."""

    f16_kv: bool = Field(False, alias="f16_kv")
    """Use half-precision for key/value cache."""

    logits_all: bool = Field(False, alias="logits_all")
    """Return logits for all tokens, not just the last token."""

    vocab_only: bool = Field(False, alias="vocab_only")
    """Only load the vocabulary, no weights."""

    use_mlock: bool = Field(False, alias="use_mlock")
    """Force system to keep model in RAM."""

    n_threads: Optional[int] = Field(None, alias="n_threads")
    """Number of threads to use. If None, the number 
    of threads is automatically determined."""

    n_batch: Optional[int] = Field(8, alias="n_batch")
    """Number of tokens to process in parallel.
    Should be a number between 1 and n_ctx."""

    n_gpu_layers: Optional[int] = Field(None, alias="n_gpu_layers")
    """Number of layers to be loaded into gpu memory. Default None."""

    verbose: bool = Field(True, alias="verbose")
    """Print verbose output to stderr."""

    def build_model(self) -> LlamaCppEmbeddings:
        model_setup = self.dict().copy()
        model_setup.pop("name")
        return LlamaCppEmbeddings(**model_setup)


EmbeddingSettingsTypes = Union[HFEmbeddingsSettings, LlamaCppSettings]
