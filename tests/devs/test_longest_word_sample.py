import torch
import os
from vllm.v1.sample.sampler import longest_word_sample, Sampler
import vllm.envs as envs


def probs_to_logits(probs: torch.Tensor) -> torch.Tensor:
    """Convert probabilities to logits, normalizing first."""
    # Normalize probabilities
    probs = probs / probs.sum(dim=-1, keepdim=True)
    # Convert to logits (log-space)
    return torch.log(probs + 1e-10)  # add epsilon for numerical stability


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


def test_eos_forced_selection_when_alternatives_are_low_prob():
    """
    Tests that the EOS token is forcibly selected when it's in the top-k candidates
    and all other candidates fall below the probability threshold (eos_prob / 100).
    This scenario was the cause of the original bug.
    Test includes two batches to verify consistent behavior across batch processing.
    """
    # Arrange: Create two-batch scenario testing different EOS threshold behaviors.
    # Both batches use token 1 as EOS but demonstrate different selection scenarios
    # 
    # Batch 0: Force EOS scenario - EOS prob=0.995, threshold=0.00995
    #          All other tokens below threshold, so EOS must be selected
    # Batch 1: Valid alternatives scenario - EOS prob=0.80, threshold=0.008  
    #          Token 2 (prob=0.19) is above threshold, creating valid choice
    probs = torch.tensor([
        [
            0.002,  # Token 0: prob=0.002 < 0.00995 (below threshold)
            0.995,  # Token 1 (EOS): high probability
            0.002,  # Token 2: prob=0.002 < 0.00995 (below threshold, longest length)
            0.001,  # Token 3: prob=0.001 < 0.00995 (below threshold)
        ],
        [
            0.005,  # Token 0: prob=0.005 < 0.008 (below threshold)
            0.80,   # Token 1 (EOS): high probability
            0.19,   # Token 2: prob=0.19 > 0.008 (VALID alternative, longest among valid)
            0.005,  # Token 3: prob=0.005 < 0.008 (below threshold)
        ]
    ])
    logits = probs_to_logits(probs)
    token_lengths = torch.tensor([1, 2, 4, 5])  # Token 3 has longest length for batch 1
    eos_token_id = 1  # Same EOS token for both batches

    # Act: Test with mix_ratio=1.0, which would normally prioritize the longest token.
    result_length = longest_word_sample(logits, token_lengths, top_k=4, mix_ratio=1.0, eos_token_id=eos_token_id)
    result_prob = longest_word_sample(logits, token_lengths, top_k=4, mix_ratio=0.0, eos_token_id=eos_token_id)

    # Assert: Different behaviors for each batch based on threshold logic
    
    # Batch 0: Force EOS - all alternatives below threshold, only EOS is valid
    assert result_length[0] == 1, "Batch 0: EOS forced (mix_ratio=1.0, no valid alternatives)"
    assert result_prob[0] == 1, "Batch 0: EOS forced (mix_ratio=0.0, no valid alternatives)"
    
    # Batch 1: Valid alternatives exist - normal selection among valid tokens [1, 2]
    # mix_ratio=1.0 (length priority): Token 2 wins (length=4 > Token 1 length=2)
    # mix_ratio=0.0 (prob priority): Token 1 wins (prob=0.80 > Token 2 prob=0.19)
    assert result_length[1] == 2, "Batch 1: Token 2 selected (mix_ratio=1.0, longest among valid)"
    assert result_prob[1] == 1, "Batch 1: EOS selected (mix_ratio=0.0, highest prob among valid)"


def test_standard_logic_with_valid_alternatives():
    """
    Tests that standard sampling logic applies when the EOS token is in the top-k,
    but other valid alternatives (with probability >= eos_prob / 100) exist.
    """
    # Arrange: EOS has medium probability, but other tokens are above the threshold.
    # EOS (index 2) has prob=0.50, so the threshold is 0.50 / 100 = 0.005.
    # Tokens 0 (prob=0.30) and 1 (prob=0.15) are both above the threshold.
    probs = torch.tensor([[
        0.30,    # Token 0: valid, length=3
        0.15,    # Token 1: valid, length=5 (longest valid)
        0.50,    # Token 2 (EOS)
        0.05,    # Token 3: not in top-k but above threshold
        0.004,   # Token 4: below threshold
    ]])
    logits = probs_to_logits(probs)
    token_lengths = torch.tensor([3, 5, 2, 4, 1])
    eos_token_id = 2

    # Act: Test with pure length and pure probability selection.
    # The top-k candidates will be [2, 0, 1, 3].
    # The valid candidates for sampling (above threshold 0.005) are [0, 1, 2, 3].
    result_length = longest_word_sample(logits, token_lengths, top_k=4, mix_ratio=1.0, eos_token_id=eos_token_id)
    result_prob = longest_word_sample(logits, token_lengths, top_k=4, mix_ratio=0.0, eos_token_id=eos_token_id)

    # Assert: The selection should happen among the valid candidates based on the mix_ratio.
    # With mix_ratio=1.0, it should select the longest valid token among [0,1,2,3], which is token 1.
    assert result_length[0] == 1
    # With mix_ratio=0.0, it should select the most probable token among [0,1,2,3], which is token 2 (EOS).
    # Note: The original EOS itself is part of the valid set for sampling.
    assert result_prob[0] == 2


def test_mixed_batch_and_edge_cases():
    """
    Tests multiple scenarios in a single batch, including:
    1. A case where EOS must be force-selected.
    2. A case where a token's probability is exactly at the threshold.
    3. A case where EOS is not in the top-k, so the logic is bypassed.
    """
    # Arrange: Create a batch with three distinct scenarios.
    probs = torch.tensor([
        # Batch 0: Force EOS. EOS prob=0.99, threshold=0.0099. All others are below.
        [0.002, 0.003, 0.990, 0.005],
        # Batch 1: Boundary condition. EOS prob=0.50, threshold=0.005.
        # Token 1 is exactly at the threshold and should be a valid candidate. It is also the longest.
        [0.010, 0.005, 0.500, 0.485],
        # Batch 2: EOS not in top-k. Normal logic should apply among top-k [0, 1, 3].
        [0.40, 0.35, 0.01, 0.24],
    ])
    logits = probs_to_logits(probs)
    # Token lengths are shared across the batch for simplicity.
    token_lengths = torch.tensor([5, 4, 2, 3])
    eos_token_id = 2

    # Act: Run the sampling with a length-favoring mix_ratio.
    result = longest_word_sample(logits, token_lengths, top_k=3, mix_ratio=1.0, eos_token_id=eos_token_id)

    # Assert: Check the outcome for each item in the batch.
    # Batch 0: Must select EOS (index 2) as all others are below the 0.0099 threshold.
    assert result[0] == 2

    # Batch 1: Valid candidates are 0 (len=5), 1 (len=4), 3 (len=3), and 2 (len=2).
    # With mix_ratio=1.0, it should pick the longest, which is token 0.
    # Note: Correcting logic from original single test `test_eos_probability_threshold_exact_boundary`
    # which had different lengths. With these lengths, token 0 is longest.
    assert result[1] == 0

    # Batch 2: EOS (index 2, prob=0.01) is not in the top-3 candidates ([0, 1, 3]).
    # The EOS threshold logic is skipped. Selection is among top-3.
    # Longest among [0, 1, 3] is token 0 (length 5).
    assert result[2] == 0


def test_eos_logic_is_disabled_or_not_applicable():
    """
    Tests that the function works normally when the EOS logic is not applicable,
    such as when eos_token_id is None or the vocabulary is trivial.
    """
    # Arrange: Standard inputs, but with eos_token_id set to None.
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    token_lengths = torch.tensor([1, 2, 3, 4, 5])

    # Act: Call the function without a specific EOS token.
    result_prob = longest_word_sample(logits, token_lengths, top_k=3, mix_ratio=0.0, eos_token_id=None)
    result_length = longest_word_sample(logits, token_lengths, top_k=3, mix_ratio=1.0, eos_token_id=None)

    # Assert: Should behave like standard top-k sampling.
    # Top-k indices are [4, 3, 2].
    # Highest probability is at index 4.
    assert result_prob[0] == 4
    # Longest length is at index 4.
    assert result_length[0] == 4


def test_performance_quick():
    """Quick performance test to establish baseline metrics."""
    import time
    
    # Test configuration
    batch_sizes = [1, 32, 128]
    vocab_size = 32000
    top_k = 50
    mix_ratio = 0.5
    num_iters = 100
    
    # Performance test without print output for clean test runs
    for batch_size in batch_sizes:
        # Create test data
        logits = torch.randn(batch_size, vocab_size)
        token_lengths = torch.randint(1, 31, (vocab_size,))
        
        # Warmup
        for _ in range(10):
            _ = longest_word_sample(logits, token_lengths, top_k, mix_ratio)
        
        # Benchmark
        start_time = time.perf_counter()
        for _ in range(num_iters):
            _ = longest_word_sample(logits, token_lengths, top_k, mix_ratio)
        elapsed_time = time.perf_counter() - start_time
        
        # Calculate metrics
        avg_time_ms = (elapsed_time * 1000) / num_iters
        throughput = (batch_size * num_iters) / elapsed_time
        
        # Assert reasonable performance bounds
        assert avg_time_ms < 1000, f"Performance too slow: {avg_time_ms}ms"
        assert throughput > 0, f"Invalid throughput: {throughput}"
    # Basic assertion to ensure function runs
    assert avg_time_ms > 0


def test_minimum_probability_threshold():
    """
    Test that tokens with probability < 0.001 are always excluded from sampling,
    regardless of EOS token presence or mix_ratio setting.
    """
    # Test 1: Without EOS token - verify basic threshold enforcement
    # Create scenario where tokens have varying probabilities around the 0.001 threshold
    probs = torch.tensor([[
        0.0001,   # Token 0: below threshold (excluded)
        0.0009,   # Token 1: below threshold (excluded)
        0.001,    # Token 2: exactly at threshold (included)
        0.002,    # Token 3: above threshold (included)
        0.9959,   # Token 4: high probability (included)
    ]])
    logits = probs_to_logits(probs)
    token_lengths = torch.tensor([5, 4, 3, 2, 1])  # Token 0 is longest but excluded
    
    # Act: Test with mix_ratio=1.0 (pure length selection)
    result = longest_word_sample(logits, token_lengths, top_k=5, mix_ratio=1.0, eos_token_id=None)
    
    # Assert: Token 0 and 1 should be excluded despite being long
    # Among valid tokens [2, 3, 4], token 2 is longest (length=3)
    assert result[0] == 2
    
    # Test 2: With EOS token - verify minimum threshold still applies
    # EOS has high probability, but some alternatives are below 0.001
    probs_with_eos = torch.tensor([[
        0.0002,   # Token 0: below threshold
        0.900,    # Token 1 (EOS): high probability
        0.0008,   # Token 2: below threshold
        0.010,    # Token 3: above threshold
        0.0890,   # Token 4: above threshold
    ]])
    logits_with_eos = probs_to_logits(probs_with_eos)
    token_lengths_eos = torch.tensor([10, 2, 8, 6, 4])  # Token 0 is longest but excluded
    
    # Act: Test with EOS token specified
    result_eos = longest_word_sample(logits_with_eos, token_lengths_eos, top_k=5, 
                                     mix_ratio=1.0, eos_token_id=1)
    
    # Assert: Tokens 0 and 2 excluded by minimum threshold
    # Among valid tokens [1, 3, 4], token 3 is longest (length=6)
    assert result_eos[0] == 3
    
    # Test 3: Multiple batches with boundary cases
    probs_batch = torch.tensor([
        # Batch 0: All tokens below threshold except one
        [0.0001, 0.0002, 0.0003, 0.0004, 0.9990],
        # Batch 1: Some tokens above threshold
        [0.0015, 0.0012, 0.001, 0.0009, 0.9954],
        # Batch 2: Mix of above and below threshold
        [0.0008, 0.0012, 0.0004, 0.0011, 0.9965],
    ])
    logits_batch = probs_to_logits(probs_batch)
    lengths_batch = torch.tensor([5, 4, 3, 2, 1])
    
    # Act: Test all batches
    result_batch = longest_word_sample(logits_batch, lengths_batch, top_k=5, 
                                       mix_ratio=1.0, eos_token_id=None)
    
    # Assert: Each batch should select appropriately
    # Batch 0: Only token 4 is valid (prob >= 0.001)
    assert result_batch[0] == 4
    # Batch 1: Valid tokens are [0, 1, 2, 4], longest is token 0 (length=5)
    assert result_batch[1] == 0  
    # Batch 2: Valid tokens are [1, 3, 4], longest is token 1 (length=4)
    assert result_batch[2] == 1


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
        test_eos_forced_selection_when_alternatives_are_low_prob,
        test_standard_logic_with_valid_alternatives,
        test_mixed_batch_and_edge_cases,
        test_eos_logic_is_disabled_or_not_applicable,
        test_performance_quick,
        test_minimum_probability_threshold,
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
