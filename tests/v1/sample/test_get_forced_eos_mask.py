# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.sample.sampler import get_forced_eos_mask
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.logits_processor import LogitsProcessorManager


def create_sampling_metadata(output_lengths: list[int]) -> SamplingMetadata:
    """Create SamplingMetadata for testing."""
    batch_size = len(output_lengths)
    output_token_ids = [[i for i in range(length)] for length in output_lengths]
    
    return SamplingMetadata(
        temperature=torch.ones(batch_size),
        all_greedy=False,
        all_random=True,
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


def test_no_eos_when_position_is_none():
    """Test that no mask is returned when eos_position is None."""
    metadata = create_sampling_metadata([5, 7, 9])
    logits = torch.randn(3, 100)
    device = torch.device("cpu")
    
    mask = get_forced_eos_mask(metadata, None, device, logits)
    assert mask is None


def test_no_eos_when_no_output_tokens():
    """Test that no mask is returned when there are no output tokens."""
    metadata = create_sampling_metadata([])
    metadata.output_token_ids = []
    logits = torch.empty(0, 100)
    device = torch.device("cpu")
    
    mask = get_forced_eos_mask(metadata, 5, device, logits)
    assert mask is None


def test_eos_forced_when_max_logit_below_threshold():
    """Test EOS is forced when max logit < threshold at positions >= eos_position."""
    metadata = create_sampling_metadata([5, 7, 8, 9])
    device = torch.device("cpu")
    
    # Create logits that result in specific max log probabilities
    logits = torch.zeros(4, 100)
    logits[0, 0] = 5.0   # pos 5, < eos_position -> no EOS regardless
    logits[1, 0] = 0.0   # pos 7, >= eos_position, logprob ≈ -4.6 < -0.7 -> force EOS
    logits[2, 0] = 0.0   # pos 8, >= eos_position, logprob ≈ -4.6 < -0.7 -> force EOS  
    logits[3, 0] = 5.0   # pos 9, >= eos_position, logprob ≈ -0.51 > -0.7 -> no EOS
    
    mask = get_forced_eos_mask(metadata, 7, device, logits, threshold_base=-0.7, threshold_coeff=0.0)
    
    expected_mask = torch.tensor([False, True, True, False])
    assert torch.equal(mask, expected_mask)


def test_eos_not_forced_when_max_logit_above_threshold():
    """Test EOS is NOT forced when max logit >= threshold even at positions >= eos_position."""
    metadata = create_sampling_metadata([5, 7, 8, 9])
    device = torch.device("cpu")
    
    # Create logits where all sequences have high confidence (max logprob > -0.7)
    logits = torch.zeros(4, 100)
    logits[:, 0] = 5.0  # This gives max logprob ≈ -0.51 > -0.7
    
    mask = get_forced_eos_mask(metadata, 7, device, logits, threshold_base=-0.7, threshold_coeff=0.0)
    
    # No sequences should be forced to EOS (high confidence)
    assert mask is None


def test_mixed_batch_with_different_thresholds():
    """Test mixed batch where only some sequences meet the threshold condition."""
    metadata = create_sampling_metadata([6, 7, 7, 8, 10])
    device = torch.device("cpu")
    
    # Create varied logits for different max log probabilities
    logits = torch.zeros(5, 100)
    logits[0, 0] = 5.0   # pos 6, < eos_position - no EOS
    logits[1, 0] = 5.0   # pos 7, >= eos_position, logprob ≈ -0.51 > -0.7 - no EOS
    logits[2, 0] = 0.0   # pos 7, >= eos_position, logprob ≈ -4.6 < -0.7 - force EOS
    logits[3, 0] = 0.0   # pos 8, >= eos_position, logprob ≈ -4.6 < -0.7 - force EOS
    logits[4, 0] = 10.0  # pos 10, >= eos_position, logprob ≈ -0.0045 > -0.7 - no EOS
    
    mask = get_forced_eos_mask(metadata, 7, device, logits, threshold_base=-0.7, threshold_coeff=0.0)
    
    expected_mask = torch.tensor([False, False, True, True, False])
    assert torch.equal(mask, expected_mask)


def test_custom_threshold():
    """Test with a custom threshold value."""
    metadata = create_sampling_metadata([5, 5, 5])
    device = torch.device("cpu")
    
    logits = torch.zeros(3, 100)
    logits[0, 0] = 10.0  # logprob ≈ -0.0045 > -0.4 - no EOS
    logits[1, 0] = 5.0   # logprob ≈ -0.51 < -0.4 - force EOS
    logits[2, 0] = 0.0   # logprob ≈ -4.6 < -0.4 - force EOS
    
    # With threshold -0.4, only the last two should get EOS
    mask = get_forced_eos_mask(metadata, 5, device, logits, threshold_base=-0.4, threshold_coeff=0.0)
    expected_mask = torch.tensor([False, True, True])
    assert torch.equal(mask, expected_mask)


def test_edge_case_exactly_at_threshold():
    """Test behavior when max logit is exactly at the threshold."""
    metadata = create_sampling_metadata([7])
    device = torch.device("cpu")
    
    # Since it's hard to get exactly -0.7, let's test the boundary behavior
    # We'll use a logit that gives logprob slightly above -0.7
    logits = torch.zeros(1, 100)
    logits[0, 0] = 4.5  # This gives logprob ≈ -0.79, which is < -0.7
    
    mask = get_forced_eos_mask(metadata, 7, device, logits, threshold_base=-0.7, threshold_coeff=0.0)
    
    # Should force EOS since logprob < threshold
    expected_mask = torch.tensor([True])
    assert torch.equal(mask, expected_mask)


def test_position_threshold_boundary():
    """Test position boundary conditions with >= eos_position."""
    metadata = create_sampling_metadata([6, 7, 8])
    device = torch.device("cpu")
    
    # All have low confidence (logprob < -0.7)
    logits = torch.zeros(3, 100)  # All logits 0 give logprob ≈ -4.6
    
    mask = get_forced_eos_mask(metadata, 7, device, logits, threshold_base=-0.7, threshold_coeff=0.0)
    
    # Only positions >= 7 should get EOS
    expected_mask = torch.tensor([False, True, True])
    assert torch.equal(mask, expected_mask)


@pytest.mark.parametrize("eos_position,output_lengths,expected_positions", [
    (5, [4, 5, 6, 7], [False, True, True, True]),
    (10, [8, 9, 10, 11], [False, False, True, True]),
    (0, [0, 1, 2], [True, True, True]),
])
def test_position_threshold_parametrized(eos_position, output_lengths, expected_positions):
    """Parametrized test for position threshold behavior."""
    metadata = create_sampling_metadata(output_lengths)
    device = torch.device("cpu")
    
    # All sequences have low confidence
    logits = torch.full((len(output_lengths), 100), -1.0)
    
    mask = get_forced_eos_mask(metadata, eos_position, device, logits, threshold_base=-0.7, threshold_coeff=0.0)
    
    expected_mask = torch.tensor(expected_positions)
    if expected_mask.any():
        assert torch.equal(mask, expected_mask)
    else:
        assert mask is None


def test_realistic_production_scenario():
    """Test scenario similar to production with high confidence."""
    metadata = create_sampling_metadata([10])
    device = torch.device("cpu")
    
    # Simulate high-confidence logits (like max_logits=20.125+ from prod)
    logits = torch.zeros(1, 100)
    logits[0, 42] = 25.0  # Extremely high confidence for one token
    
    mask = get_forced_eos_mask(metadata, 10, device, logits, threshold_base=-0.7, threshold_coeff=0.0)
    
    # Should NOT force EOS due to high confidence
    assert mask is None


def test_dynamic_threshold_based_on_length():
    """Test dynamic threshold that changes based on sequence length."""
    metadata = create_sampling_metadata([10, 50, 100, 200])
    device = torch.device("cpu")
    
    # Create logits for different behaviors based on dynamic thresholds
    logits = torch.zeros(4, 100)
    logits[0, 0] = 5.0   # logprob ≈ -0.51, threshold@10 = -0.7  -> -0.51 > -0.7 (no EOS)
    logits[1, 0] = 4.2   # logprob ≈ -0.96, threshold@50 = -0.66 -> -0.96 < -0.66 (force EOS)
    logits[2, 0] = 5.5   # logprob ≈ -0.42, threshold@100 = -0.61 -> -0.42 > -0.61 (no EOS)
    logits[3, 0] = 2.0   # logprob ≈ -2.66, threshold@200 = -0.51 -> -2.66 < -0.51 (force EOS)
    
    mask = get_forced_eos_mask(metadata, 10, device, logits, threshold_base=-0.71, threshold_coeff=0.001)
    
    expected_mask = torch.tensor([False, True, False, True])
    assert torch.equal(mask, expected_mask)