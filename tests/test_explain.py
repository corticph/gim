"""Tests for gim.explain() function."""
import pytest
import torch


class TestExplain:
    """Tests for the explain function."""

    def test_import(self):
        """Test that explain can be imported."""
        from gim import explain
        assert explain is not None

    def test_explain_basic(self, tiny_model, sample_tokens):
        """Test basic explain functionality."""
        from gim import explain

        attributions = explain(tiny_model, sample_tokens, baseline_token_id=0)

        assert attributions.shape == (sample_tokens.shape[1],)
        assert attributions.dtype == torch.float32

    def test_explain_requires_baseline_or_tokenizer(self, tiny_model, sample_tokens):
        """Test that explain raises error without baseline_token_id or tokenizer."""
        from gim import explain

        with pytest.raises(ValueError, match="Either baseline_token_id or tokenizer"):
            explain(tiny_model, sample_tokens)

    def test_explain_1d_input(self, tiny_model):
        """Test explain with 1D input tensor."""
        from gim import explain

        tokens_1d = torch.randint(0, 1000, (8,))
        attributions = explain(tiny_model, tokens_1d, baseline_token_id=0)

        assert attributions.shape == (8,)

    def test_explain_2d_input(self, tiny_model, sample_tokens):
        """Test explain with 2D input tensor."""
        from gim import explain

        attributions = explain(tiny_model, sample_tokens, baseline_token_id=0)

        assert attributions.shape == (sample_tokens.shape[1],)

    def test_explain_with_target_token(self, tiny_model, sample_tokens):
        """Test explain with specific target token."""
        from gim import explain

        target_token = 42
        attributions = explain(
            tiny_model, sample_tokens,
            baseline_token_id=0,
            target_token_id=target_token
        )

        assert attributions.shape == (sample_tokens.shape[1],)

    def test_explain_different_targets_differ(self, tiny_model, sample_tokens):
        """Test that different target tokens produce different attributions."""
        from gim import explain

        attr_42 = explain(tiny_model, sample_tokens, baseline_token_id=0, target_token_id=42)
        attr_100 = explain(tiny_model, sample_tokens, baseline_token_id=0, target_token_id=100)

        assert not torch.allclose(attr_42, attr_100)

    def test_explain_target_position(self, tiny_model, sample_tokens):
        """Test explain at different target positions."""
        from gim import explain

        attr_last = explain(tiny_model, sample_tokens, baseline_token_id=0, target_position=-1)
        attr_mid = explain(tiny_model, sample_tokens, baseline_token_id=0, target_position=4)

        assert attr_last.shape == attr_mid.shape
        assert not torch.allclose(attr_last, attr_mid)

    def test_explain_custom_gim_params(self, tiny_model, sample_tokens):
        """Test explain with custom GIM parameters."""
        from gim import explain

        attr_default = explain(tiny_model, sample_tokens, baseline_token_id=0)
        attr_custom = explain(
            tiny_model, sample_tokens,
            baseline_token_id=0,
            freeze_norm=True,
            softmax_temperature=4.0,
            q_scale=0.1,
            k_scale=0.1,
            v_scale=0.8,
        )

        assert not torch.allclose(attr_default, attr_custom)

    def test_explain_gim_disabled(self, tiny_model, sample_tokens):
        """Test explain with GIM disabled."""
        from gim import explain

        attr_gim = explain(tiny_model, sample_tokens, baseline_token_id=0)
        attr_no_gim = explain(
            tiny_model, sample_tokens,
            baseline_token_id=0,
            freeze_norm=False,
            softmax_temperature=None,
            q_scale=None,
            k_scale=None,
            v_scale=None,
        )

        # Should produce different results
        assert not torch.allclose(attr_gim, attr_no_gim)

    def test_explain_non_trivial(self, tiny_model, sample_tokens):
        """Test that attributions are non-trivial (not all zeros)."""
        from gim import explain

        attributions = explain(tiny_model, sample_tokens, baseline_token_id=0)

        assert (attributions != 0).any(), "Attributions should not all be zero"

    def test_explain_unsupported_model_type(self, sample_tokens):
        """Test that explain raises error for unsupported model types."""
        from gim import explain

        with pytest.raises(TypeError, match="Unsupported model type"):
            explain("not a model", sample_tokens, baseline_token_id=0)
