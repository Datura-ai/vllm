import torch
import os
from vllm.v1.sample.sampler import longest_word_sample, Sampler
import vllm.envs as envs


def setup_function():
    """Set random seed for deterministic tests."""
    torch.manual_seed(42)


def test_pure_length_selection():
    """Test that mix_ratio=1.0 selects longest tokens from top-k."""
    # Arrange: Create logits where highest probabilities are at shorter tokens
    logits = torch.tensor([
        [0.1, 0.2, 0.3, 0.4, 0.5],  # highest prob at index 4
        [0.5, 0.4, 0.3, 0.2, 0.1],  # highest prob at index 0
    ])
    token_lengths = torch.tensor([5, 4, 3, 2, 1])  # longest at index 0
    
    # Act: Use pure length selection
    result = longest_word_sample(logits, token_lengths, top_k=3, mix_ratio=1.0)
    
    # Assert: For batch 0, top-3 by prob are [4,3,2], longest among them is index 2 (length=3)
    # For batch 1, top-3 by prob are [0,1,2], longest among them is index 0 (length=5)
    assert result[0] == 2
    assert result[1] == 0


def test_pure_probability_selection():
    """Test that mix_ratio=0.0 selects highest probability tokens from top-k."""
    # Arrange: Create logits where shortest tokens have highest probabilities
    logits = torch.tensor([
        [0.5, 0.4, 0.3, 0.2, 0.1],  # highest prob at index 0
        [0.1, 0.2, 0.3, 0.4, 0.5],  # highest prob at index 4
    ])
    token_lengths = torch.tensor([1, 2, 3, 4, 5])  # shortest at index 0
    
    # Act: Use pure probability selection
    result = longest_word_sample(logits, token_lengths, top_k=3, mix_ratio=0.0)
    
    # Assert: For batch 0, top-3 by prob are [0,1,2], highest prob among them is index 0
    # For batch 1, top-3 by prob are [4,3,2], highest prob among them is index 4
    assert result[0] == 0
    assert result[1] == 4


def test_mixed_scoring():
    """Test that mix_ratio=0.5 balances probability and length."""
    # Arrange: Create scenario where pure length and pure probability give different results
    logits = torch.tensor([[0.1, 0.4, 0.2, 0.3]])  # highest prob at index 1
    token_lengths = torch.tensor([1, 2, 3, 4])  # longest at index 3
    
    # Act: Test different mix ratios
    result_pure_prob = longest_word_sample(logits, token_lengths, top_k=4, mix_ratio=0.0)
    result_mixed = longest_word_sample(logits, token_lengths, top_k=4, mix_ratio=0.5)
    result_pure_length = longest_word_sample(logits, token_lengths, top_k=4, mix_ratio=1.0)
    
    # Assert: Pure probability should select index 1 (highest prob)
    # Pure length should select index 3 (longest)
    # Mixed should potentially select a different token that balances both
    assert result_pure_prob[0] == 1
    assert result_pure_length[0] == 3
    # Mixed result should be valid but may differ from pure strategies
    assert 0 <= result_mixed[0] < 4


def test_mixed_scoring_calculated():
    """Test mixed scoring with manually calculated expected results."""
    # Arrange: Simple case with known mixed scores
    logits = torch.tensor([[0.8, 0.6, 0.4, 0.2]])  # decreasing probabilities
    token_lengths = torch.tensor([1, 2, 3, 4])  # increasing lengths
    
    # Act: Use 50/50 mix
    result = longest_word_sample(logits, token_lengths, top_k=4, mix_ratio=0.5)
    
    # Assert: Calculate expected mixed scores manually
    # topk_probs = [0.8, 0.6, 0.4, 0.2]
    # topk_lengths = [1, 2, 3, 4]
    # normalized_lengths = [0.25, 0.5, 0.75, 1.0]
    # mixed_scores = 0.5 * [0.8, 0.6, 0.4, 0.2] + 0.5 * [0.25, 0.5, 0.75, 1.0]
    # mixed_scores = [0.525, 0.55, 0.575, 0.6]
    # highest mixed score is at index 3
    assert result[0] == 3


def test_edge_case_large_top_k():
    """Test that top_k larger than vocab_size is handled correctly."""
    # Arrange: Small vocabulary
    logits = torch.tensor([[0.1, 0.2, 0.3]])
    token_lengths = torch.tensor([1, 2, 3])
    
    # Act: Use top_k larger than vocab_size
    result = longest_word_sample(logits, token_lengths, top_k=10, mix_ratio=1.0)
    
    # Assert: Should work with effective top_k = vocab_size = 3
    # Longest token among all is index 2
    assert result[0] == 2


def test_edge_case_top_k_one():
    """Test behavior with top_k=1."""
    # Arrange: Multiple tokens but only top-1 considered
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    token_lengths = torch.tensor([10, 1, 2, 3])
    
    # Act: Use top_k=1
    result = longest_word_sample(logits, token_lengths, top_k=1, mix_ratio=1.0)
    
    # Assert: Only top-1 by probability (index 3) is considered
    # Since only one token in top-k, it must be selected regardless of mix_ratio
    assert result[0] == 3


def test_edge_case_equal_lengths():
    """Test behavior when all tokens have equal lengths."""
    # Arrange: All tokens have same length
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    token_lengths = torch.tensor([5, 5, 5, 5])
    
    # Act: Use pure length selection
    result = longest_word_sample(logits, token_lengths, top_k=3, mix_ratio=1.0)
    
    # Assert: When all lengths are equal, normalized lengths are all 1.0
    # With mix_ratio=1.0, all top-k tokens have same mixed score
    # Result should be one of the top-3 tokens [3, 2, 1]
    assert result[0] in [1, 2, 3]


def test_edge_case_equal_probabilities():
    """Test behavior when all logits are equal."""
    # Arrange: All logits equal
    logits = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    token_lengths = torch.tensor([1, 2, 3, 4])
    
    # Act: Use pure length selection
    result = longest_word_sample(logits, token_lengths, top_k=3, mix_ratio=1.0)
    
    # Assert: When probabilities are equal, top-k selection is arbitrary
    # But among selected tokens, longest should be chosen
    # Result should be a valid token index
    assert 0 <= result[0] < 4


def test_batch_consistency():
    """Test that identical batches produce identical results."""
    # Arrange: Create identical batches
    logits = torch.tensor([
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.1, 0.2, 0.3, 0.4, 0.5],
    ])
    token_lengths = torch.tensor([1, 2, 3, 4, 5])
    
    # Act: Process both batches
    result = longest_word_sample(logits, token_lengths, top_k=3, mix_ratio=0.7)
    
    # Assert: Both batches should produce same result
    assert result[0] == result[1]


def test_env_variable_loading():
    """Test that VLLM_MIX_RATIO environment variable is loaded correctly."""
    # Arrange: Set environment variable
    original_value = os.environ.get('VLLM_MIX_RATIO')
    os.environ['VLLM_MIX_RATIO'] = '0.3'
    
    try:
        # Act: Import environment variable (reload module to pick up change)
        import importlib
        importlib.reload(envs)
        
        # Assert: Environment variable should be loaded as float
        assert envs.VLLM_MIX_RATIO == 0.3
        
    finally:
        # Cleanup: Restore original value
        if original_value is not None:
            os.environ['VLLM_MIX_RATIO'] = original_value
        else:
            os.environ.pop('VLLM_MIX_RATIO', None)


def test_sampler_class_integration():
    """Test that Sampler class correctly uses mix_ratio from environment."""
    # Arrange: Set environment variable
    original_value = os.environ.get('VLLM_MIX_RATIO')
    os.environ['VLLM_MIX_RATIO'] = '0.8'
    
    try:
        # Act: Create Sampler instance
        import importlib
        importlib.reload(envs)
        
        token_lengths_gpu = torch.tensor([1, 2, 3, 4, 5])
        sampler = Sampler(token_lengths_gpu)
        
        # Assert: Sampler should use environment variable value
        assert sampler.mix_ratio == 0.8
        
    finally:
        # Cleanup: Restore original value
        if original_value is not None:
            os.environ['VLLM_MIX_RATIO'] = original_value
        else:
            os.environ.pop('VLLM_MIX_RATIO', None)


def test_deterministic_behavior():
    """Test that function produces deterministic results with same inputs."""
    # Arrange: Create test data
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    token_lengths = torch.tensor([1, 2, 3, 4, 5])
    
    # Act: Call function multiple times with same inputs
    torch.manual_seed(42)
    result1 = longest_word_sample(logits, token_lengths, top_k=3, mix_ratio=0.5)
    
    torch.manual_seed(42)
    result2 = longest_word_sample(logits, token_lengths, top_k=3, mix_ratio=0.5)
    
    # Assert: Results should be identical
    assert torch.equal(result1, result2)


def test_eos_token_prioritization():
    """Test that EOS token is prioritized when present in top-k."""
    # Arrange: Create logits where EOS token has high probability
    logits = torch.tensor([
        [0.1, 0.2, 0.9, 0.3, 0.4],  # EOS at index 2, highest prob
        [0.5, 0.4, 0.1, 0.2, 0.3],  # EOS at index 2, low prob
    ])
    token_lengths = torch.tensor([1, 2, 3, 4, 5])
    eos_token_id = 2
    
    # Act: Use different mix ratios
    result_prob = longest_word_sample(logits, token_lengths, top_k=3, mix_ratio=0.0, eos_token_id=eos_token_id)
    result_mixed = longest_word_sample(logits, token_lengths, top_k=3, mix_ratio=0.5, eos_token_id=eos_token_id)
    result_length = longest_word_sample(logits, token_lengths, top_k=3, mix_ratio=1.0, eos_token_id=eos_token_id)
    
    # Assert: Batch 0 should always select EOS (index 2) since it's in top-k
    assert result_prob[0] == 2
    assert result_mixed[0] == 2
    assert result_length[0] == 2
    
    # Batch 1 should use normal logic since EOS is not in top-k
    assert result_prob[1] == 0  # highest prob among top-k
    assert result_length[1] == 4  # longest among top-k


def test_eos_token_not_in_topk():
    """Test that normal logic works when EOS token is not in top-k."""
    # Arrange: Create logits where EOS token has very low probability
    logits = torch.tensor([[0.5, 0.4, 0.01, 0.3, 0.2]])  # EOS at index 2, very low prob
    token_lengths = torch.tensor([1, 2, 3, 4, 5])
    eos_token_id = 2
    
    # Act: Use pure length selection
    result = longest_word_sample(logits, token_lengths, top_k=3, mix_ratio=1.0, eos_token_id=eos_token_id)
    
    # Assert: Should select longest token among top-k [0,1,3], which is index 3
    assert result[0] == 3
    assert result[0] != eos_token_id


def test_eos_token_mixed_batch():
    """Test batch where some requests have EOS in top-k and others don't."""
    # Arrange: Create mixed scenario
    logits = torch.tensor([
        [0.1, 0.2, 0.9, 0.3, 0.4],  # EOS at index 2, high prob (in top-k)
        [0.5, 0.4, 0.01, 0.3, 0.2],  # EOS at index 2, low prob (not in top-k)
        [0.2, 0.1, 0.8, 0.3, 0.4],  # EOS at index 2, high prob (in top-k)
        [0.4, 0.5, 0.02, 0.3, 0.2],  # EOS at index 2, low prob (not in top-k)
    ])
    token_lengths = torch.tensor([1, 2, 3, 4, 5])
    eos_token_id = 2
    
    # Act: Use mixed ratio
    result = longest_word_sample(logits, token_lengths, top_k=3, mix_ratio=0.5, eos_token_id=eos_token_id)
    
    # Assert: Batches 0 and 2 should select EOS token
    assert result[0] == eos_token_id
    assert result[2] == eos_token_id
    
    # Batches 1 and 3 should use normal logic
    assert result[1] != eos_token_id
    assert result[3] != eos_token_id


def test_eos_token_none():
    """Test that function works normally when eos_token_id is None."""
    # Arrange: Create test data
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    token_lengths = torch.tensor([1, 2, 3, 4, 5])
    
    # Act: Use different mix ratios with no EOS token
    result_prob = longest_word_sample(logits, token_lengths, top_k=3, mix_ratio=0.0, eos_token_id=None)
    result_length = longest_word_sample(logits, token_lengths, top_k=3, mix_ratio=1.0, eos_token_id=None)
    
    # Assert: Should behave normally
    assert result_prob[0] == 4  # highest prob among top-k [4,3,2]
    assert result_length[0] == 4  # longest among top-k [4,3,2] with lengths [5,4,3]


def test_eos_token_edge_cases():
    """Test EOS token behavior in edge cases."""
    # Arrange: Single token vocabulary with EOS
    logits = torch.tensor([[0.5]])
    token_lengths = torch.tensor([3])
    eos_token_id = 0
    
    # Act: Test with top_k=1
    result = longest_word_sample(logits, token_lengths, top_k=1, mix_ratio=0.5, eos_token_id=eos_token_id)
    
    # Assert: Should select the only token (which is EOS)
    assert result[0] == 0
    
    # Test with EOS token ID out of range
    logits = torch.tensor([[0.1, 0.2, 0.3]])
    token_lengths = torch.tensor([1, 2, 3])
    eos_token_id = 5  # Out of range
    
    result = longest_word_sample(logits, token_lengths, top_k=2, mix_ratio=1.0, eos_token_id=eos_token_id)
    
    # Assert: Should work normally since EOS is not in vocabulary
    assert result[0] == 2  # longest among top-k [2,1] with lengths [3,2]


if __name__ == "__main__":
    # Run tests manually since pytest may not be available
    import sys
    
    test_functions = [
        test_pure_length_selection,
        test_pure_probability_selection,
        test_mixed_scoring,
        test_mixed_scoring_calculated,
        test_edge_case_large_top_k,
        test_edge_case_top_k_one,
        test_edge_case_equal_lengths,
        test_edge_case_equal_probabilities,
        test_batch_consistency,
        test_env_variable_loading,
        test_sampler_class_integration,
        test_deterministic_behavior,
        test_eos_token_prioritization,
        test_eos_token_not_in_topk,
        test_eos_token_mixed_batch,
        test_eos_token_none,
        test_eos_token_edge_cases,
    ]
    
    print("Running longest_word_sample tests...")
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            setup_function()
            test_func()
            print(f"✓ {test_func.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__}: {e}")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
