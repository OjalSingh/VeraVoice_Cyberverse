import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def tiny_embedder():
    """A randomly initialised 2-layer WavLM, so tests run offline in seconds."""
    import torch
    from transformers import WavLMConfig, WavLMModel

    from veravoice.embed import WavLMEmbedder

    torch.manual_seed(0)
    cfg = WavLMConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        conv_dim=(32, 32, 32, 32, 32, 32, 32),
        num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=2,
    )
    return WavLMEmbedder(backbone="tiny-test", device="cpu", model=WavLMModel(cfg))
