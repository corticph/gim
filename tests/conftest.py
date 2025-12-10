"""Pytest fixtures for GIM tests."""
import pytest
import torch

from tests.models import TinyLM


@pytest.fixture
def tiny_model():
    """Create a small transformer model for testing."""
    model = TinyLM(vocab_size=1000, d_model=128, n_layers=2, n_heads=4)
    model.eval()
    return model


@pytest.fixture
def sample_tokens():
    """Create sample token tensors for testing."""
    return torch.randint(0, 1000, (1, 8))
