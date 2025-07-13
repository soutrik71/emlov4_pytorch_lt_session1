"""Sample test file to verify pytest setup."""

import lightning
import pytest
import torch


def test_pytorch_installation():
    """Test that PyTorch is properly installed."""
    assert torch.__version__ is not None
    print(f"PyTorch version: {torch.__version__}")


def test_lightning_installation():
    """Test that Lightning is properly installed."""
    assert lightning.__version__ is not None
    print(f"Lightning version: {lightning.__version__}")


def test_mps_availability():
    """Test MPS (Apple Silicon GPU) availability."""
    if torch.backends.mps.is_available():
        print("MPS is available for GPU acceleration!")
    else:
        print("MPS is not available")


def test_basic_tensor_operations():
    """Test basic PyTorch tensor operations."""
    # Create tensors
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])

    # Basic operations
    z = x + y
    expected = torch.tensor([5.0, 7.0, 9.0])

    assert torch.allclose(z, expected)


def test_tensor_on_mps():
    """Test tensor operations on MPS device if available."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        x = torch.tensor([1.0, 2.0, 3.0]).to(device)
        y = torch.tensor([4.0, 5.0, 6.0]).to(device)
        z = x + y

        expected = torch.tensor([5.0, 7.0, 9.0]).to(device)
        assert torch.allclose(z, expected)
        print("MPS tensor operations working correctly!")
    else:
        pytest.skip("MPS not available")


@pytest.mark.parametrize("input_size", [10, 100, 1000])
def test_tensor_creation_sizes(input_size):
    """Test tensor creation with different sizes."""
    tensor = torch.randn(input_size)
    assert tensor.shape == (input_size,)
    assert tensor.dtype == torch.float32


class TestTensorOperations:
    """Test class for tensor operations."""

    def test_matrix_multiplication(self):
        """Test matrix multiplication."""
        a = torch.randn(3, 4)
        b = torch.randn(4, 5)
        c = torch.mm(a, b)

        assert c.shape == (3, 5)

    def test_tensor_reshape(self):
        """Test tensor reshaping."""
        x = torch.randn(12)
        y = x.view(3, 4)

        assert y.shape == (3, 4)
        assert x.numel() == y.numel()
