# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.sample.sampler import Sampler
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.logits_processor import LogitsProcessorManager


def create_mock_sampling_metadata(output_lengths: torch.Tensor) -> SamplingMetadata:
    """Create a minimal SamplingMetadata for testing"""
    batch_size = len(output_lengths)
    output_token_ids = [list(range(length)) for length in output_lengths.tolist()]
    
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


def test_eos_forced_with_very_low_logits():
    """Test case where EOS should be forced due to very low max logits"""
    token_lengths = torch.zeros(100, dtype=torch.int32)
    sampler = Sampler(
        token_lengths_gpu=token_lengths,
        eos_token_id=2,
        eos_position=5
    )
    
    # Create logits where max value is below threshold
    logits = torch.zeros(3, 100)
    logits[0, :] = -1.0   # Max logit = -1.0, below threshold -0.7, pos 5 >= 5 -> force EOS
    logits[1, :] = -0.8   # Max logit = -0.8, below threshold -0.7, pos 6 >= 5 -> force EOS  
    logits[2, :] = 0.5    # Max logit = 0.5, above threshold -0.7, pos 7 >= 5 -> no EOS
    
    metadata = create_mock_sampling_metadata(torch.tensor([5, 6, 7]))
    result = sampler.sample(logits, metadata)
    
    # First two should be EOS, third should not be
    assert result[0] == 2, f"Expected EOS for low logits at pos 5, got {result[0]}"
    assert result[1] == 2, f"Expected EOS for low logits at pos 6, got {result[1]}"
    assert result[2] != 2, f"Should not force EOS for high logits at pos 7, got {result[2]}"


def test_mixed_position_confidence_scenario():
    """Test mixed scenario with various positions and confidence levels"""
    token_lengths = torch.zeros(100, dtype=torch.int32)
    sampler = Sampler(
        token_lengths_gpu=token_lengths,
        eos_token_id=2,
        eos_position=8
    )
    
    # Test various combinations of position and confidence
    logits = torch.zeros(5, 100)
    logits[0, :] = -1.0   # pos 7, < 8, low confidence -> no EOS (wrong position)
    logits[1, :] = -1.0   # pos 8, >= 8, low confidence -> force EOS
    logits[2, :] = 2.0    # pos 9, >= 8, high confidence -> no EOS
    logits[3, :] = -0.8   # pos 10, >= 8, low confidence -> force EOS
    logits[4, :] = -0.6   # pos 11, >= 8, marginal confidence -> no EOS (above threshold)
    
    metadata = create_mock_sampling_metadata(torch.tensor([7, 8, 9, 10, 11]))
    result = sampler.sample(logits, metadata)
    
    # Expected: [no EOS, EOS, no EOS, EOS, no EOS]
    assert result[0] != 2, "Position 7 < 8, should not force EOS"
    assert result[1] == 2, f"Position 8 >= 8 with low confidence, should force EOS, got {result[1]}"
    assert result[2] != 2, "Position 9 >= 8 but high confidence, should not force EOS"
    assert result[3] == 2, f"Position 10 >= 8 with low confidence, should force EOS, got {result[3]}"
    assert result[4] != 2, "Position 11 >= 8 but confidence above threshold, should not force EOS"


def test_production_high_confidence_scenario():
    """Test production-like scenario with very high confidence"""
    token_lengths = torch.zeros(100, dtype=torch.int32)
    sampler = Sampler(
        token_lengths_gpu=token_lengths,
        eos_token_id=2,
        eos_position=10
    )
    
    # Simulate high-confidence logits like in production (max_logits=25.0)
    logits = torch.zeros(1, 100)
    logits[0, 42] = 25.0  # Extremely high confidence for one token
    
    metadata = create_mock_sampling_metadata(torch.tensor([10]))
    result = sampler.sample(logits, metadata)
    
    # Should NOT force EOS due to high confidence
    assert result[0] != 2, "High confidence should not force EOS"
    assert result[0] == 42, "Should select the high-confidence token"


def test_edge_cases_with_eos_position():
    """Test edge cases around eos_position boundary"""
    token_lengths = torch.zeros(100, dtype=torch.int32)
    sampler = Sampler(
        token_lengths_gpu=token_lengths,
        eos_token_id=99,
        eos_position=5
    )
    
    # All sequences have low confidence
    logits = torch.full((4, 100), -1.0)
    
    metadata = create_mock_sampling_metadata(torch.tensor([4, 5, 6, 7]))
    result = sampler.sample(logits, metadata)
    
    # Only positions >= 5 should get EOS
    assert result[0] != 99, "Position 4 < 5, should not force EOS"
    assert result[1] == 99, "Position 5 >= 5 with low confidence, should force EOS"
    assert result[2] == 99, "Position 6 >= 5 with low confidence, should force EOS"
    assert result[3] == 99, "Position 7 >= 5 with low confidence, should force EOS"


@pytest.mark.parametrize("eos_position", [3, 10, 0])
def test_parametrized_eos_forcing(eos_position):
    """Parametrized test for different EOS positions with low confidence"""
    token_lengths = torch.zeros(50, dtype=torch.int32)
    sampler = Sampler(
        token_lengths_gpu=token_lengths,
        eos_token_id=1,
        eos_position=eos_position
    )
    
    # Create batch with positions before, at, and after eos_position
    test_positions = [eos_position - 1, eos_position, eos_position + 1]
    test_positions = [max(0, pos) for pos in test_positions]  # Ensure non-negative
    
    # Low confidence logits (below default threshold -0.7)
    logits = torch.full((len(test_positions), 50), -1.0)
    
    metadata = create_mock_sampling_metadata(torch.tensor(test_positions))
    result = sampler.sample(logits, metadata)
    
    for i, pos in enumerate(test_positions):
        if pos >= eos_position:
            assert result[i] == 1, f"Position {pos} >= {eos_position}, should force EOS"
        else:
            assert result[i] != 1, f"Position {pos} < {eos_position}, should not force EOS"


def test_no_eos_forcing_when_disabled():
    """Test that EOS is not forced when eos_position is None"""
    token_lengths = torch.zeros(100, dtype=torch.int32)
    sampler = Sampler(
        token_lengths_gpu=token_lengths,
        eos_token_id=2,
        eos_position=None  # Disabled
    )
    
    # Even with very low confidence, should not force EOS
    logits = torch.full((2, 100), -10.0)
    
    metadata = create_mock_sampling_metadata(torch.tensor([10, 20]))
    result = sampler.sample(logits, metadata)
    
    # Should not force EOS regardless of confidence
    assert result[0] != 2, "Should not force EOS when eos_position is None"
    assert result[1] != 2, "Should not force EOS when eos_position is None"