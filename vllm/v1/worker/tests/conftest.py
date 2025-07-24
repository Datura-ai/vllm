"""Test fixtures for worker module tests."""

from dataclasses import dataclass
from typing import Optional
from unittest.mock import Mock, patch

import pytest
import torch


@dataclass
class MockModelConfig:
    """Mock ModelConfig for testing precompute_token_lengths function."""
    tokenizer: str
    trust_remote_code: bool = False
    tokenizer_revision: Optional[str] = None


@pytest.fixture
def deepseek_config() -> MockModelConfig:
    """Configuration for DeepSeek model."""
    return MockModelConfig(
        tokenizer="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        trust_remote_code=False,
        tokenizer_revision=None
    )


@pytest.fixture  
def qwen_config() -> MockModelConfig:
    """Configuration for Qwen model."""
    return MockModelConfig(
        tokenizer="Qwen/QwQ-32B", 
        trust_remote_code=False,
        tokenizer_revision=None
    )


@pytest.fixture
def llama_tokenizer_fix_config() -> MockModelConfig:
    """Configuration for Llama tokenizer fix model."""
    return MockModelConfig(
        tokenizer="tau-vision/llama-tokenizer-fix",
        trust_remote_code=False, 
        tokenizer_revision=None
    )


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer with predefined vocabulary."""
    tokenizer = Mock()
    tokenizer.__len__ = Mock(return_value=1000)  # Small vocab for testing
    tokenizer.eos_token_id = 2  # Mock EOS token ID
    
    # Mock decode method to return predictable strings
    def mock_decode(token_ids, skip_special_tokens=True):
        token_id = token_ids[0] if isinstance(token_ids, list) else token_ids
        if token_id == 0:
            return ""  # BOS/PAD token
        elif token_id == 1:
            return "a"  # Single character
        elif token_id == 2:
            return "hello"  # Multi-character word
        elif token_id == 999:
            raise Exception("Invalid token")  # Simulating decode error
        else:
            return f"token_{token_id}"  # Generic token
    
    tokenizer.decode = Mock(side_effect=mock_decode)
    return tokenizer


@pytest.fixture
def patch_auto_tokenizer(mock_tokenizer):
    """Patch AutoTokenizer.from_pretrained to return mock tokenizer."""
    with patch('vllm.v1.worker.gpu_model_runner.AutoTokenizer.from_pretrained') as mock_from_pretrained:
        mock_from_pretrained.return_value = mock_tokenizer
        yield mock_from_pretrained


@pytest.fixture
def no_tqdm():
    """Disable tqdm progress bars in tests."""
    with patch('vllm.v1.worker.gpu_model_runner.tqdm') as mock_tqdm:
        mock_tqdm.side_effect = lambda x, **kwargs: x
        yield mock_tqdm


@pytest.fixture
def mock_bad_tokens():
    """Mock get_bad_tokens_by_length function."""
    with patch('vllm.v1.worker.gpu_model_runner.get_bad_tokens_by_length') as mock_func:
        mock_func.return_value = [500, 501]  # Mock bad token IDs
        yield mock_func


@pytest.fixture
def mock_envs():
    """Mock environment variables."""
    with patch('vllm.v1.worker.gpu_model_runner.envs') as mock_envs:
        mock_envs.VLLM_BAD_TOKENS_IDS = None
        mock_envs.VLLM_BAD_TOKENS_ALL = False
        mock_envs.VLLM_EOS_TOKEN_LENGTH = 3
        yield mock_envs


@pytest.fixture
def real_deepseek_config() -> MockModelConfig:
    """Real configuration for DeepSeek model."""
    return MockModelConfig(
        tokenizer="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        trust_remote_code=False,
        tokenizer_revision=None
    )


@pytest.fixture  
def real_qwen_config() -> MockModelConfig:
    """Real configuration for Qwen model."""
    return MockModelConfig(
        tokenizer="Qwen/QwQ-32B", 
        trust_remote_code=False,
        tokenizer_revision=None
    )


@pytest.fixture
def real_llama_tokenizer_fix_config() -> MockModelConfig:
    """Real configuration for Llama tokenizer fix model."""
    return MockModelConfig(
        tokenizer="tau-vision/llama-tokenizer-fix",
        trust_remote_code=False, 
        tokenizer_revision=None
    )


@pytest.fixture
def real_mistral_config() -> MockModelConfig:
    """Real configuration for Mistral model."""
    return MockModelConfig(
        tokenizer="casperhansen/mistral-nemo-instruct-2407-awq",
        trust_remote_code=False,
        tokenizer_revision=None
    )


@pytest.fixture
def mock_bad_tokens_real():
    """Mock get_bad_tokens_by_length function for real tests."""
    with patch('vllm.v1.worker.gpu_model_runner.get_bad_tokens_by_length') as mock_func:
        mock_func.return_value = []  # No bad tokens for real tests
        yield mock_func