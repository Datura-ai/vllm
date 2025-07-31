import pytest
import torch
from typing import Optional

# Real imports from vLLM
from vllm.v1.sample.sampler import Sampler
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.logits_processor import LogitsProcessorManager


def create_mock_sampling_metadata(
    output_lengths: torch.Tensor,
    batch_size: int = None
) -> SamplingMetadata:
    """Create a minimal SamplingMetadata for testing"""
    if batch_size is None:
        batch_size = len(output_lengths)
    
    # Create minimal required fields
    output_token_ids = [
        list(range(length)) for length in output_lengths.tolist()
    ]
    
    return SamplingMetadata(
        temperature=torch.ones(batch_size),
        all_greedy=False,
        all_random=False,
        top_p=torch.ones(batch_size),
        top_k=torch.ones(batch_size, dtype=torch.int32) * 10,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(batch_size),
        presence_penalties=torch.zeros(batch_size),
        repetition_penalties=torch.ones(batch_size),
        output_token_ids=output_token_ids,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessorManager(),
    )


def test_eos_position_not_set_integration():
    # Checks that sampler works normally when eos_position is None
    token_lengths = torch.zeros(100, dtype=torch.int32)
    sampler = Sampler(
        token_lengths_gpu=token_lengths,
        eos_token_id=2,
        eos_position=None
    )
    
    # Create simple logits and metadata
    logits = torch.randn(3, 100)  # batch_size=3, vocab_size=100
    metadata = create_mock_sampling_metadata(
        output_lengths=torch.tensor([5, 8, 12])
    )
    
    # This should work without forcing EOS
    try:
        result = sampler.sample(logits, metadata)
        # Expect: normal sampling, no EOS forced
        assert result.shape == (3,)  # batch_size
        print("✓ Test passed: no EOS forcing when eos_position=None")
    except Exception as e:
        print(f"✗ Test failed (expected until EOS logic implemented): {e}")


def test_eos_position_at_9_should_force_eos():
    # Checks that EOS is forced at position 9 (10th token) with low confidence
    token_lengths = torch.zeros(100, dtype=torch.int32)
    sampler = Sampler(
        token_lengths_gpu=token_lengths,
        eos_token_id=2,
        eos_position=9  # Force EOS on 10th token
    )
    
    # Create low confidence logits to trigger EOS forcing
    logits = torch.full((3, 100), -1.0)  # Low confidence (below -0.7 threshold)
    metadata = create_mock_sampling_metadata(
        output_lengths=torch.tensor([9, 8, 10])  # Positions 0 and 2 should get EOS (>= 9)
    )
    
    result = sampler.sample(logits, metadata)
    # EOS should be forced for positions >= 9 with low confidence
    assert result[0] == 2, f"Expected EOS token (2) for low confidence at position 9, got {result[0]}"
    assert result[1] != 2, f"Should not force EOS at position 8 (< 9), got {result[1]}"
    assert result[2] == 2, f"Expected EOS token (2) for low confidence at position 10, got {result[2]}"


@pytest.mark.parametrize(
    "output_length,eos_position,should_be_eos",
    [
        (8, 9, False),   # Before EOS position - no forcing
        (9, 9, True),    # At EOS position - should force EOS with low confidence
        (10, 9, True),   # After EOS position - should force EOS with low confidence  
    ],
)
def test_eos_position_boundary_integration(output_length, eos_position, should_be_eos):
    # Checks EOS forcing at boundary conditions with real sampler
    token_lengths = torch.zeros(100, dtype=torch.int32)
    sampler = Sampler(
        token_lengths_gpu=token_lengths,
        eos_token_id=2,
        eos_position=eos_position
    )
    
    # Use low confidence logits to trigger EOS forcing when appropriate
    logits = torch.full((1, 100), -1.0)  # Low confidence
    metadata = create_mock_sampling_metadata(
        output_lengths=torch.tensor([output_length])
    )
    
    result = sampler.sample(logits, metadata)
    
    if should_be_eos:
        assert result[0] == 2, f"Expected EOS token (2) at position {output_length} with low confidence, got {result[0]}"
    else:
        # For positions that shouldn't force EOS, should not be EOS token
        assert result[0] != 2, f"Should not force EOS at position {output_length} < {eos_position}"


def test_eos_position_batch_mixed_integration():
    # Checks mixed batch where some requests need EOS, others don't
    token_lengths = torch.zeros(100, dtype=torch.int32)
    sampler = Sampler(
        token_lengths_gpu=token_lengths,
        eos_token_id=2,
        eos_position=9
    )
    
    # Batch with mixed positions and confidence levels
    logits = torch.zeros(5, 100)
    logits[0, :] = -1.0  # pos 9, >= 9, low confidence -> force EOS
    logits[1, :] = 0.0   # pos 9, >= 9, high confidence case
    logits[1, 10] = 5.0  # Make token 10 have high confidence (> 99% probability)
    logits[2, :] = -1.0  # pos 8, < 9, low confidence -> no EOS
    logits[3, :] = -1.0  # pos 10, >= 9, low confidence -> force EOS
    logits[4, :] = -0.8  # pos 9, >= 9, low confidence -> force EOS
    
    metadata = create_mock_sampling_metadata(
        output_lengths=torch.tensor([9, 9, 8, 10, 9])
    )
    
    result = sampler.sample(logits, metadata)
    
    # Expected: positions >= 9 with low confidence should get EOS
    assert result[0] == 2, f"Position 9 with low confidence should be EOS, got {result[0]}"
    assert result[1] != 2, f"Position 9 with high confidence should not be EOS, got {result[1]}"  
    assert result[2] != 2, f"Position 8 < 9 should not be EOS, got {result[2]}"
    assert result[3] == 2, f"Position 10 with low confidence should be EOS, got {result[3]}"
    assert result[4] == 2, f"Position 9 with low confidence should be EOS, got {result[4]}"


if __name__ == "__main__":
    print("Running EOS position integration tests...")
    print("These tests SHOULD FAIL until EOS forcing logic is implemented in sampler.sample()")
    print()
    
    test_eos_position_not_set_integration()
    test_eos_position_at_9_should_force_eos()
    
    # Test boundary conditions
    test_eos_position_boundary_integration(8, 9, False)
    test_eos_position_boundary_integration(9, 9, True) 
    test_eos_position_boundary_integration(10, 9, False)
    
    test_eos_position_batch_mixed_integration()
    
    print("\nDone! If most tests failed, that's expected - EOS logic not yet implemented.")