"""Tests for GIM context manager."""
import pytest
import torch
import torch.nn.functional as F


class TestGIMContext:
    """Tests for the GIM context manager."""

    def test_import(self):
        """Test that GIM can be imported."""
        from gim import GIM
        assert GIM is not None

    def test_gim_default_params(self, tiny_model, sample_tokens):
        """Test GIM with default parameters."""
        from gim import GIM

        x = sample_tokens[:, :-1]
        y = sample_tokens[:, 1:]

        with GIM(tiny_model):
            logits = tiny_model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1)
            )
            loss.backward()

        # Check that gradients exist
        assert tiny_model.blocks[0].attn.Wq.weight.grad is not None
        assert tiny_model.blocks[0].attn.Wk.weight.grad is not None
        assert tiny_model.blocks[0].attn.Wv.weight.grad is not None

    def test_gim_modifies_gradients(self, tiny_model, sample_tokens):
        """Test that GIM modifies gradient magnitudes."""
        from gim import GIM

        x = sample_tokens[:, :-1]
        y = sample_tokens[:, 1:]

        # First, get gradients without GIM
        logits = tiny_model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        grad_q_no_gim = tiny_model.blocks[0].attn.Wq.weight.grad.norm().item()
        tiny_model.zero_grad(set_to_none=True)

        # Now with GIM
        with GIM(tiny_model):
            logits = tiny_model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
        grad_q_with_gim = tiny_model.blocks[0].attn.Wq.weight.grad.norm().item()

        # Gradients should be different
        assert grad_q_no_gim != grad_q_with_gim

    def test_gim_disabled_matches_no_gim(self, tiny_model, sample_tokens):
        """Test that GIM with all options disabled matches no GIM."""
        from gim import GIM

        x = sample_tokens[:, :-1]
        y = sample_tokens[:, 1:]

        # Without GIM
        logits = tiny_model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        grad_q_no_gim = tiny_model.blocks[0].attn.Wq.weight.grad.clone()
        tiny_model.zero_grad(set_to_none=True)

        # With GIM disabled
        with GIM(tiny_model, freeze_norm=False, softmax_temperature=None,
                 q_scale=None, k_scale=None, v_scale=None):
            logits = tiny_model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
        grad_q_disabled = tiny_model.blocks[0].attn.Wq.weight.grad.clone()

        # Should be approximately equal
        assert torch.allclose(grad_q_no_gim, grad_q_disabled, rtol=1e-4)

    def test_gim_restores_state(self, tiny_model, sample_tokens):
        """Test that GIM properly restores model state after context exit."""
        from gim import GIM

        # Get original norm modules
        original_ln1 = tiny_model.blocks[0].ln1

        with GIM(tiny_model):
            pass

        # After exiting, should be restored
        assert tiny_model.blocks[0].ln1 is original_ln1

    def test_gim_custom_scales(self, tiny_model, sample_tokens):
        """Test GIM with custom Q/K/V scales."""
        from gim import GIM

        x = sample_tokens[:, :-1]
        y = sample_tokens[:, 1:]

        with GIM(tiny_model, q_scale=0.1, k_scale=0.1, v_scale=0.8):
            logits = tiny_model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()

        assert tiny_model.blocks[0].attn.Wq.weight.grad is not None
