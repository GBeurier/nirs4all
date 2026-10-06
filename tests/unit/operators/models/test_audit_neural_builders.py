"""Declared neural pooling and activation parameters must affect execution."""

import pytest


def test_torch_pooling_matches_declared_tokens():
    torch = pytest.importorskip("torch")
    from nirs4all.operators.models.pytorch.spectral_transformer import SpectralTransformer

    model = SpectralTransformer((1, 32), embed_dim=8, depth=0, num_heads=2, patch_size=8,
                                dropout=0, drop_path=0, pool="mean")
    model.eval()
    x = torch.arange(64, dtype=torch.float32).reshape(2, 1, 32) / 64
    tokens = model.patch_embed(x)
    tokens = torch.cat([model.cls_token.expand(2, -1, -1), tokens], dim=1)
    tokens = model.norm(model.pos_embed(tokens))
    expected = model.head(tokens[:, 1:].mean(dim=1))
    with torch.no_grad():
        assert torch.allclose(model(x), expected)
        model.pool = "cls"
        assert torch.allclose(model(x), model.head(tokens[:, 0]))
    with pytest.raises(ValueError, match="pool"):
        SpectralTransformer((1, 32), pool="other")


def test_torch_identity_activation_and_unknown_names():
    torch = pytest.importorskip("torch")
    from nirs4all.operators.models.pytorch.nicon import get_activation

    x = torch.tensor([-1., 2.])
    assert torch.equal(get_activation(None)(x), x)
    with pytest.raises(ValueError, match="Unknown activation"):
        get_activation("misspelled")


def test_jax_identity_activation_and_unknown_names():
    pytest.importorskip("flax")
    from nirs4all.operators.models.jax.nicon import get_activation

    marker = object()
    assert get_activation(None)(marker) is marker
    with pytest.raises(ValueError, match="Unknown activation"):
        get_activation("misspelled")
