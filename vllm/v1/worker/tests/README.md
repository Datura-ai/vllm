# Tests for vLLM v1 Worker Module

## Overview

This directory contains tests for the `vllm.v1.worker` module, specifically focusing on the `precompute_token_lengths` function in `gpu_model_runner.py`.

## Test Structure

### Files
- `test_gpu_model_runner.py` - Main test file for GPU model runner functionality
- `conftest.py` - Test fixtures and configuration  
- `__init__.py` - Package initialization

### Test Functions

Tests are organized as individual functions following pytest best practices (one test = one function):

#### General Functionality Tests
- `test_precompute_token_lengths_basic_functionality` - Basic functionality and return types
- `test_precompute_token_lengths_correct_values` - Correct token length computation
- `test_precompute_token_lengths_handles_decode_errors` - Error handling during token decoding
- `test_precompute_token_lengths_different_models` - Parametrized test for all three models
- `test_precompute_token_lengths_respects_trust_remote_code` - Parameter passing validation
- `test_precompute_token_lengths_respects_tokenizer_revision` - Parameter passing validation
- `test_precompute_token_lengths_tensor_properties` - Tensor properties validation

#### Model-Specific Tests
- `test_deepseek_model_specific_functionality` - DeepSeek tokenizer behavior
- `test_deepseek_token_length_distribution` - Token length distribution analysis
- `test_qwen_model_specific_functionality` - Qwen tokenizer behavior
- `test_qwen_large_vocab_handling` - Large vocabulary handling
- `test_llama_tokenizer_fix_functionality` - Llama tokenizer fix behavior
- `test_llama_tokenizer_fix_special_handling` - Special token handling

#### Bad Token Handling Tests
- `test_bad_tokens_get_forced_length` - Bad tokens get forced length of 5
- `test_eos_token_gets_forced_length` - EOS token handling
- `test_get_bad_tokens_called_correctly` - Integration with `get_bad_tokens_by_length`

#### Integration Tests with Real Tokenizers
- `test_real_deepseek_tokenizer_functionality` - Real DeepSeek tokenizer integration
- `test_real_qwen_tokenizer_functionality` - Real Qwen tokenizer integration  
- `test_real_llama_tokenizer_functionality` - Real Llama tokenizer integration
- `test_real_tokenizers_vocab_sizes` - Parametrized test for vocabulary sizes
- `test_real_tokenizers_specific_tokens` - Specific token handling with real tokenizers

## Running Tests

```bash
# Run all tests (unit + integration)
python -m pytest vllm/v1/worker/tests/test_gpu_model_runner.py -v

# Run only unit tests (fast, with mocks)
python -m pytest vllm/v1/worker/tests/test_gpu_model_runner.py -v -m "not integration"

# Run only integration tests (slow, with real tokenizers)
python -m pytest vllm/v1/worker/tests/test_gpu_model_runner.py -v -m integration

# Run specific test
python -m pytest vllm/v1/worker/tests/test_gpu_model_runner.py::test_precompute_token_lengths_basic_functionality -v

# Run real tokenizer tests
python -m pytest vllm/v1/worker/tests/test_gpu_model_runner.py::test_real_deepseek_tokenizer_functionality -v
```

## Test Design Principles

Following project best practices:
- **AAA Pattern**: Arrange → Act → Assert
- **Atomic Tests**: One assertion per test concept
- **Parametrized Tests**: Using `@pytest.mark.parametrize` instead of loops
- **Proper Comments**: Brief description at top, explanation before assertions
- **Fast Execution**: All external dependencies mocked
- **Deterministic**: Same results on every run

## Test Types

### Unit Tests (Fast)
Use comprehensive mocking for fast, reliable execution:
- `AutoTokenizer.from_pretrained` → Returns mock tokenizer
- `get_bad_tokens_by_length` → Returns predefined bad token IDs
- Environment variables → Controlled test values
- `tqdm` → Disabled for clean test output

### Integration Tests (Slow)
Use real tokenizers from HuggingFace:
- Downloads real model tokenizers on first run
- Tests actual vocabulary sizes and token behaviors
- Verifies real-world integration
- Marked with `@pytest.mark.integration`

## Coverage

Tests cover:
- ✅ Basic functionality across all three models
- ✅ Parameter handling and validation
- ✅ Error conditions and edge cases  
- ✅ Bad token processing logic
- ✅ Token length computation accuracy
- ✅ Integration with external dependencies
- ✅ Real tokenizer behavior and vocabulary sizes
- ✅ Actual model-specific token handling