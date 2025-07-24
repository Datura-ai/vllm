"""Tests for gpu_model_runner module, focusing on precompute_token_lengths function."""

import torch
import pytest
from unittest.mock import Mock

from vllm.v1.worker.gpu_model_runner import precompute_token_lengths


def test_precompute_token_lengths_basic_functionality(deepseek_config, patch_auto_tokenizer, no_tqdm, mock_bad_tokens, mock_envs):
    # Checks basic functionality of precompute_token_lengths
    result = precompute_token_lengths(deepseek_config)
    
    # Expect torch.Tensor with correct dtype and shape
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.int32
    assert result.shape == (1000,)  # Mock vocab size


def test_precompute_token_lengths_correct_values(deepseek_config, patch_auto_tokenizer, no_tqdm, mock_bad_tokens, mock_envs):
    # Checks that token lengths are computed correctly
    result = precompute_token_lengths(deepseek_config)
    
    # Expect specific lengths based on mock tokenizer decode results
    assert result[0] == 0   # Empty string
    assert result[1] == 1   # Single character "a"
    assert result[2] == 3   # EOS token gets forced length of 3


def test_precompute_token_lengths_handles_decode_errors(deepseek_config, patch_auto_tokenizer, no_tqdm, mock_bad_tokens, mock_envs):
    # Checks error handling during token decoding
    result = precompute_token_lengths(deepseek_config)
    
    # Expect zero length for tokens that fail to decode (token 999)
    assert result[999] == 0


@pytest.mark.parametrize(
    "config_fixture,expected_model_name",
    [
        ("deepseek_config", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"),
        ("qwen_config", "Qwen/QwQ-32B"),
        ("llama_tokenizer_fix_config", "tau-vision/llama-tokenizer-fix"),
    ],
)
def test_precompute_token_lengths_different_models(config_fixture, expected_model_name, request, patch_auto_tokenizer, no_tqdm, mock_bad_tokens, mock_envs):
    # Checks function works with different model configurations
    config = request.getfixturevalue(config_fixture)
    
    result = precompute_token_lengths(config)
    
    # Expect AutoTokenizer.from_pretrained called with correct model name
    patch_auto_tokenizer.assert_called_once_with(
        expected_model_name,
        trust_remote_code=False,
        tokenizer_revision=None,
    )
    # Expect valid tensor output
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.int32


def test_precompute_token_lengths_respects_trust_remote_code(deepseek_config, patch_auto_tokenizer, no_tqdm, mock_bad_tokens, mock_envs):
    # Checks trust_remote_code parameter is passed correctly
    deepseek_config.trust_remote_code = True
    
    precompute_token_lengths(deepseek_config)
    
    # Expect trust_remote_code=True passed to AutoTokenizer
    patch_auto_tokenizer.assert_called_once_with(
        "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        trust_remote_code=True,
        tokenizer_revision=None,
    )


def test_precompute_token_lengths_respects_tokenizer_revision(deepseek_config, patch_auto_tokenizer, no_tqdm, mock_bad_tokens, mock_envs):
    # Checks tokenizer_revision parameter is passed correctly
    deepseek_config.tokenizer_revision = "v1.0"
    
    precompute_token_lengths(deepseek_config)
    
    # Expect tokenizer_revision="v1.0" passed to AutoTokenizer
    patch_auto_tokenizer.assert_called_once_with(
        "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        trust_remote_code=False,
        tokenizer_revision="v1.0",
    )


def test_precompute_token_lengths_tensor_properties(deepseek_config, patch_auto_tokenizer, no_tqdm, mock_bad_tokens, mock_envs):
    # Checks output tensor has correct properties
    result = precompute_token_lengths(deepseek_config)
    
    # Expect tensor with correct properties
    assert result.dtype == torch.int32
    assert result.dim() == 1
    assert result.min() >= 0  # No negative lengths
    assert result.max() < 1000  # Reasonable upper bound for token lengths


def test_deepseek_model_specific_functionality(deepseek_config, patch_auto_tokenizer, no_tqdm, mock_bad_tokens, mock_envs):
    # Checks DeepSeek model-specific tokenizer behavior
    result = precompute_token_lengths(deepseek_config)
    
    # Expect correct model name used
    patch_auto_tokenizer.assert_called_once_with(
        "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        trust_remote_code=False,
        tokenizer_revision=None,
    )
    # Expect valid tensor for DeepSeek tokenizer
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] > 0


def test_deepseek_token_length_distribution(deepseek_config, patch_auto_tokenizer, no_tqdm, mock_bad_tokens, mock_envs):
    # Checks token length distribution for DeepSeek model
    result = precompute_token_lengths(deepseek_config)
    
    # Expect reasonable distribution of token lengths
    assert torch.sum(result == 0) > 0  # Some zero-length tokens (special tokens)
    assert torch.sum(result > 0) > 0   # Some non-zero length tokens
    assert torch.sum(result > 10) == 0 # No extremely long tokens in mock


def test_qwen_model_specific_functionality(qwen_config, patch_auto_tokenizer, no_tqdm, mock_bad_tokens, mock_envs):
    # Checks Qwen model-specific tokenizer behavior
    result = precompute_token_lengths(qwen_config)
    
    # Expect correct model name used for Qwen
    patch_auto_tokenizer.assert_called_once_with(
        "Qwen/QwQ-32B",
        trust_remote_code=False,
        tokenizer_revision=None,
    )
    # Expect valid tensor for Qwen tokenizer
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.int32


def test_qwen_large_vocab_handling(qwen_config, patch_auto_tokenizer, no_tqdm, mock_bad_tokens, mock_envs):
    # Checks handling of large vocabulary for Qwen model
    result = precompute_token_lengths(qwen_config)
    
    # Expect tensor covers full vocabulary range
    assert result.shape[0] == 1000  # Mock vocab size
    assert torch.all(result >= 0)   # All lengths non-negative


def test_llama_tokenizer_fix_functionality(llama_tokenizer_fix_config, patch_auto_tokenizer, no_tqdm, mock_bad_tokens, mock_envs):
    # Checks Llama tokenizer fix model behavior
    result = precompute_token_lengths(llama_tokenizer_fix_config)
    
    # Expect correct model name used for Llama tokenizer fix
    patch_auto_tokenizer.assert_called_once_with(
        "tau-vision/llama-tokenizer-fix",
        trust_remote_code=False,
        tokenizer_revision=None,
    )
    # Expect valid tensor for Llama tokenizer fix
    assert isinstance(result, torch.Tensor)
    assert result.shape == (1000,)


def test_llama_tokenizer_fix_special_handling(llama_tokenizer_fix_config, patch_auto_tokenizer, no_tqdm, mock_bad_tokens, mock_envs):
    # Checks special token handling for Llama tokenizer fix
    result = precompute_token_lengths(llama_tokenizer_fix_config)
    
    # Expect consistent token length computation
    assert result[0] == 0  # BOS/special token
    assert result[1] == 1  # Single character
    assert result[2] == 3  # EOS token gets forced length of 3


def test_bad_tokens_get_forced_length(deepseek_config, patch_auto_tokenizer, no_tqdm, mock_bad_tokens, mock_envs):
    # Checks that bad tokens get forced length of 5
    result = precompute_token_lengths(deepseek_config)
    
    # Expect bad tokens (500, 501) to have forced length of 5
    assert result[500] == 5
    assert result[501] == 5


def test_eos_token_gets_forced_length(deepseek_config, patch_auto_tokenizer, no_tqdm, mock_bad_tokens, mock_envs):
    # Checks that EOS token gets forced length
    result = precompute_token_lengths(deepseek_config)
    
    # Expect EOS token (token_id=2) to have forced length of 3
    assert result[2] == 3


def test_get_bad_tokens_called_correctly(deepseek_config, patch_auto_tokenizer, no_tqdm, mock_bad_tokens, mock_envs):
    # Checks that get_bad_tokens_by_length is called with correct parameters
    precompute_token_lengths(deepseek_config)
    
    # Expect get_bad_tokens_by_length called with correct parameters
    mock_bad_tokens.assert_called_once_with(
        patch_auto_tokenizer.return_value,
        max_len=8,
        bad_token_ids=[],
        find_all=False
    )


# Integration tests with real tokenizers
@pytest.mark.integration
def test_real_deepseek_tokenizer_functionality(real_deepseek_config, no_tqdm, mock_bad_tokens_real, mock_envs):
    # Checks real DeepSeek tokenizer integration
    result = precompute_token_lengths(real_deepseek_config)

    # Expect valid tensor with real vocab size
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.int32
    assert result.shape[0] > 50000  # DeepSeek has large vocabulary
    assert result.min() >= 0  # No negative lengths
    

@pytest.mark.integration
def test_real_qwen_tokenizer_functionality(real_qwen_config, no_tqdm, mock_bad_tokens_real, mock_envs):
    # Checks real Qwen tokenizer integration
    result = precompute_token_lengths(real_qwen_config)
    
    # Expect valid tensor with real vocab size
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.int32
    assert result.shape[0] > 50000  # Qwen has large vocabulary
    assert result.min() >= 0  # No negative lengths


@pytest.mark.integration
def test_real_llama_tokenizer_functionality(real_llama_tokenizer_fix_config, no_tqdm, mock_bad_tokens_real, mock_envs):
    # Checks real Llama tokenizer fix integration
    result = precompute_token_lengths(real_llama_tokenizer_fix_config)
    
    # Expect valid tensor with real vocab size
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.int32
    assert result.shape[0] > 30000  # Llama has substantial vocabulary
    assert result.min() >= 0  # No negative lengths


@pytest.mark.integration
def test_real_mistral_tokenizer_functionality(real_mistral_config, no_tqdm, mock_bad_tokens_real, mock_envs):
    # Checks real Mistral tokenizer integration
    mock_envs.VLLM_EOS_TOKEN_LENGTH = 1
    result = precompute_token_lengths(real_mistral_config)
    
    # Expect valid tensor with real vocab size
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.int32
    assert result.shape[0] > 30000  # Mistral has substantial vocabulary
    assert result.min() >= 0  # No negative lengths
    assert result[2] == 1  # EOS token length should be 1 as per mock_envs


@pytest.mark.integration
@pytest.mark.parametrize(
    "config_fixture,min_vocab_size",
    [
        ("real_deepseek_config", 50000),
        ("real_qwen_config", 50000),
        ("real_llama_tokenizer_fix_config", 30000),
        ("real_mistral_config", 30000),
    ],
)
def test_real_tokenizers_vocab_sizes(config_fixture, min_vocab_size, request, no_tqdm, mock_bad_tokens_real, mock_envs):
    # Checks that real tokenizers have expected vocabulary sizes
    config = request.getfixturevalue(config_fixture)
    
    result = precompute_token_lengths(config)
    
    # Expect vocabulary size meets minimum threshold
    assert result.shape[0] >= min_vocab_size
    # Expect reasonable token length distribution
    assert torch.sum(result == 0) > 0  # Some zero-length tokens (special)
    assert torch.sum(result > 0) > 0   # Some non-zero length tokens
    assert torch.sum(result > 200) == 0  # No extremely long tokens (allow some long sequences)


@pytest.mark.integration
def test_real_tokenizers_specific_tokens(real_deepseek_config, no_tqdm, mock_bad_tokens_real, mock_envs):
    # Checks specific token handling with real tokenizer
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(
        real_deepseek_config.tokenizer,
        trust_remote_code=real_deepseek_config.trust_remote_code,
        tokenizer_revision=real_deepseek_config.tokenizer_revision,
    )
    
    result = precompute_token_lengths(real_deepseek_config)
    
    # Expect EOS token has forced length of 3
    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
        assert result[tokenizer.eos_token_id] == 3  # Forced length
