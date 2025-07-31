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
    
    # Create logits where positions >= 7 have different max logits
    logits = torch.zeros(4, 100)
    logits[0, :] = 1.0   # pos 5, < eos_position, high logits - no EOS
    logits[1, :] = -1.0  # pos 7, >= eos_position, low logits - force EOS
    logits[2, :] = -0.8  # pos 8, >= eos_position, low logits - force EOS
    logits[3, :] = 0.5   # pos 9, >= eos_position, high logits - no EOS
    
    mask = get_forced_eos_mask(metadata, 7, device, logits, threshold=-0.7)
    
    expected_mask = torch.tensor([False, True, True, False])
    assert torch.equal(mask, expected_mask)


def test_eos_not_forced_when_max_logit_above_threshold():
    """Test EOS is NOT forced when max logit >= threshold even at positions >= eos_position."""
    metadata = create_sampling_metadata([5, 7, 8, 9])
    device = torch.device("cpu")
    
    # Create logits where all have high max values
    logits = torch.zeros(4, 100)
    logits[:, :] = -0.5  # Above threshold of -0.7
    
    mask = get_forced_eos_mask(metadata, 7, device, logits, threshold=-0.7)
    
    # No sequences should be forced to EOS (high confidence)
    assert mask is None


def test_mixed_batch_with_different_thresholds():
    """Test mixed batch where only some sequences meet the threshold condition."""
    metadata = create_sampling_metadata([6, 7, 7, 8, 10])
    device = torch.device("cpu")
    
    # Create varied logits
    logits = torch.zeros(5, 100)
    logits[0, :] = 0.0   # pos 6, < eos_position - no EOS
    logits[1, :] = -0.6  # pos 7, >= eos_position, above threshold - no EOS
    logits[2, :] = -0.9  # pos 7, >= eos_position, below threshold - force EOS
    logits[3, :] = -1.5  # pos 8, >= eos_position, below threshold - force EOS
    logits[4, :] = -0.2  # pos 10, >= eos_position, above threshold - no EOS
    
    mask = get_forced_eos_mask(metadata, 7, device, logits, threshold=-0.7)
    
    expected_mask = torch.tensor([False, False, True, True, False])
    assert torch.equal(mask, expected_mask)


def test_custom_threshold():
    """Test with a custom threshold value."""
    metadata = create_sampling_metadata([5, 5, 5])
    device = torch.device("cpu")
    
    logits = torch.zeros(3, 100)
    logits[0, :] = -0.3
    logits[1, :] = -0.5
    logits[2, :] = -0.7
    
    # With threshold -0.4, only the last two should get EOS
    mask = get_forced_eos_mask(metadata, 5, device, logits, threshold=-0.4)
    expected_mask = torch.tensor([False, True, True])
    assert torch.equal(mask, expected_mask)


def test_edge_case_exactly_at_threshold():
    """Test behavior when max logit is exactly at the threshold."""
    metadata = create_sampling_metadata([7])
    device = torch.device("cpu")
    
    logits = torch.zeros(1, 100)
    logits[0, :] = -0.7  # Exactly at threshold
    
    mask = get_forced_eos_mask(metadata, 7, device, logits, threshold=-0.7)
    
    # Should NOT force EOS (< threshold, not <=)
    assert mask is None


def test_position_threshold_boundary():
    """Test position boundary conditions with >= eos_position."""
    metadata = create_sampling_metadata([6, 7, 8])
    device = torch.device("cpu")
    
    # All have low confidence but different positions relative to threshold
    logits = torch.full((3, 100), -1.0)  # All below threshold
    
    mask = get_forced_eos_mask(metadata, 7, device, logits, threshold=-0.7)
    
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
    
    mask = get_forced_eos_mask(metadata, eos_position, device, logits, threshold=-0.7)
    
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
    
    mask = get_forced_eos_mask(metadata, 10, device, logits, threshold=-0.7)
    
    # Should NOT force EOS due to high confidence
    assert mask is None