from __future__ import annotations

from types import SimpleNamespace

from torch import nn

from medical_rag_reranker.models import reranker_module


class _FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.eval()


def test_cross_encoder_sets_loaded_encoder_to_train(monkeypatch) -> None:
    fake_encoder = _FakeEncoder()

    monkeypatch.setattr(
        reranker_module.AutoModel,
        "from_pretrained",
        lambda *_args, **_kwargs: fake_encoder,
    )

    model = reranker_module.CrossEncoderReranker("fake-model")

    assert model.encoder.training is True
