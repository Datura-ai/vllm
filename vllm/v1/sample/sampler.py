# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A layer that samples the next tokens from the model's outputs."""

import torch
import torch.nn as nn
from typing import Optional

from vllm.logger import init_logger
from vllm.utils import is_pin_memory_available
from vllm.v1.outputs import LogprobsTensors, SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.bad_words import apply_bad_words
from vllm.v1.sample.ops.penalties import apply_all_penalties
from vllm.v1.sample.ops.topk_topp_sampler import (
    TopKTopPSampler,
    apply_top_k_top_p,
)
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
import torch.nn.functional as F

def longest_word_sample(
        logits: torch.Tensor,
        token_lengths: torch.Tensor,
        top_k: int = 10
) -> torch.Tensor:
    """
    Sample tokens by selecting the longest word from top-k candidates.

    Args:
        logits: Tensor of shape [batch_size, vocab_size]
        token_lengths: Tensor of shape [vocab_size] with character lengths
        top_k: Number of top candidates to consider

    Returns:
        Tensor of shape [batch_size] - selected token indices
    """
    vocab_size = logits.size(-1)
    top_k = min(top_k, vocab_size)
    topk_logits, topk_indices = torch.topk(logits, k=top_k, dim=-1)  # [batch_size, top_k]

    # Get lengths for the top-k tokens

    topk_lengths = token_lengths[torch.clamp(topk_indices, 0, token_lengths.size(0) - 1)]  # [batch_size, top_k]

    # Find indices of longest tokens within each top-k set
    longest_indices_in_topk = torch.argmax(topk_lengths, dim=-1, keepdim=True)  # [batch_size, 1]

    # Use gather to select the corresponding token indices
    final_tokens = torch.gather(topk_indices, dim=-1, index=longest_indices_in_topk).squeeze(-1)  # [batch_size]

    # print("starting")
    # logger.info(f"--- [LOG: longest_word_sample] ---")
    # logger.info(f"Input data: logits.shape={list(logits.shape)}, top_k={top_k}")
    # logger.info(f"Detailed breakdown for each sequence in the batch:")
    #
    # logprobs = logits.log_softmax(dim=-1, dtype=torch.float32)
    # topk_logprobs = logprobs.gather(-1, topk_indices)  # [batch_size, top_k]
    #
    # batch_size = logits.size(0)
    # for batch_idx in range(batch_size):
    #     logger.info(f"Input data: logits.shape={list(logits.shape)}, top_k={top_k}")
    #     logger.info(f"  [Sequence {batch_idx + 1}/{batch_size}]")
    #     logger.info(f"    - Top-{top_k} candidates:")
    #
    #     for i in range(top_k):
    #         token_id = topk_indices[batch_idx, i].item()
    #         length = topk_lengths[batch_idx, i].item()
    #         logit_val = topk_logits[batch_idx, i].item()
    #         logprob_val = topk_logprobs[batch_idx, i].item()
    #         chosen = " <<< CHOSEN" if i == longest_indices_in_topk[batch_idx].item() else ""
    #         logger.info(f"      - Candidate {i + 1}: ID={token_id}, Length={length}{chosen}, logit_val={logit_val}, logprob_val={logprob_val:.4f}")
    #
    #     final_token = final_tokens[batch_idx].item()
    #     logger.info(f"    - Result: Final chosen token ID = {final_token}")

    return final_tokens




def get_param(
    sampling_metadata: SamplingMetadata,
    param_name: str,
    default_value,
    req_index: int = 0
):
    """Get parameter value with priority: per-request > legacy > default.
    
    Args:
        sampling_metadata: Sampling metadata containing parameters
        param_name: Name of parameter to retrieve
        default_value: Default value if parameter not found
        req_index: Request index for per-request parameters
        
    Returns:
        Parameter value with proper type conversion
    """
    # Priority 1: Per-request parameters
    if (sampling_metadata.extra_args_per_request and 
        req_index in sampling_metadata.extra_args_per_request):
        req_extra_args = sampling_metadata.extra_args_per_request[req_index]
        if param_name in req_extra_args:
            value = req_extra_args[param_name]
            # Convert 0/1 to boolean for boolean parameters
            if isinstance(default_value, bool):
                return bool(value)
            return value
    
    # Priority 2: Legacy batch-level parameters
    if sampling_metadata.extra_args and param_name in sampling_metadata.extra_args:
        value = sampling_metadata.extra_args[param_name]
        if isinstance(default_value, bool):
            return bool(value)
        return value
    
    # Priority 3: Default value
    return default_value


def get_forced_eos_mask(
    sampling_metadata: SamplingMetadata,
    eos_position: Optional[int],
    device: torch.device,
    logits: torch.Tensor,
    threshold_base: float = -0.71,
    threshold_coeff: float = 0.001,
) -> Optional[torch.Tensor]:
    """Return mask where EOS should be forced based on max log probability threshold, or None.
    
    Forces EOS when:
    - Sequence length >= eos_position AND 
    - Max log probability < dynamic threshold

    Dynamic threshold = threshold_base + threshold_coeff * sequence_length
    Default: -0.71 + 0.001 * length (e.g., -0.7 at 10 chars, -0.61 at 100 chars)
    """
    if eos_position is None:
        logger.info("get_forced_eos_mask: eos_position is None, no mask applied")
        return None
        
    if not sampling_metadata.output_token_ids:
        logger.info("get_forced_eos_mask: no output token ids, no mask applied")
        return None
        
    output_lengths = torch.tensor([len(tokens) for tokens in sampling_metadata.output_token_ids], device=device)
    
    # Use per-request eos_position if available, otherwise use default
    batch_size = len(sampling_metadata.output_token_ids)
    eos_positions = []
    
    for req_index in range(batch_size):
        req_eos_position = get_param(sampling_metadata, "eos_position", eos_position, req_index)
        eos_positions.append(req_eos_position)
    
    # Create per-request position masks
    position_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for i, req_eos_pos in enumerate(eos_positions):
        if req_eos_pos is not None:
            position_mask[i] = output_lengths[i] >= req_eos_pos
    
    if not position_mask.any():
        logger.info(f"get_forced_eos_mask: no sequences meet eos position criteria")
        return None
    
    # Convert logits to log probabilities
    logprobs = logits.log_softmax(dim=-1, dtype=torch.float32)
    
    # Get max log probability for each sequence
    max_logprobs, _ = torch.max(logprobs, dim=-1)
    
    # Calculate dynamic threshold for each sequence based on its length
    dynamic_thresholds = threshold_base + threshold_coeff * output_lengths.float()

    # Apply EOS only where position >= eos_position AND max logprob < dynamic threshold
    force_eos_mask = position_mask & (max_logprobs < dynamic_thresholds)
    result = force_eos_mask if force_eos_mask.any() else None
    
    # logger.info(f"get_forced_eos_mask: eos_positions={eos_positions}, lengths={output_lengths.tolist()}, "
    #             f"position_mask={position_mask.tolist()}, max_logprobs={max_logprobs.tolist()}, "
    #             f"dynamic_thresholds={dynamic_thresholds.tolist()}, "
    #             f"threshold_base={threshold_base}, threshold_coeff={threshold_coeff}, "
    #             f"final_mask={force_eos_mask.tolist()}, "
    #             f"result={'applied' if result is not None else 'none'}")
    return result


_SAMPLING_EPS = 1e-5
logger = init_logger(__name__)

class Sampler(nn.Module):

    def __init__(self, token_lengths_gpu, eos_token_id: int, eos_position: Optional[int] = None):
        super().__init__()
        self.topk_topp_sampler = TopKTopPSampler()
        self.pin_memory = is_pin_memory_available()
        self.token_lengths_gpu = token_lengths_gpu
        self.eos_token_id = eos_token_id
        self.eos_position = eos_position
        logger.info(f"starting sampler with {self.eos_token_id=} and {self.eos_position=}")

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        # NOTE(woosuk): Use the original logits (before any penalties or
        # temperature scaling) for the top-k logprobs.
        # This is different from the V0 sampler, which uses the logits that
        # is used for sampling (after penalties and temperature scaling).
        # TODO(rob): provide option for logprobs post sampling.
        # See https://vllm-dev.slack.com/archives/C07UUL8E61Z/p1735907856007919 # noqa: E501
        num_logprobs = sampling_metadata.max_num_logprobs
        if num_logprobs is not None:
            raw_logprobs = self.compute_logprobs(logits)

        # Use float32 for the logits.
        logits = logits.to(torch.float32)
        # Apply allowed token ids.
        logits = self.apply_allowed_token_ids(logits, sampling_metadata)
        # Apply bad words exclusion.
        logits = self.apply_bad_words(logits, sampling_metadata)

        # Apply logits processors which can impact greedy sampling
        for processor in (sampling_metadata.logitsprocs.non_argmax_invariant):
            logits = processor.apply(logits)

        # Apply penalties (e.g., min_tokens, freq_penalties).
        logits = self.apply_penalties(logits, sampling_metadata)
        # Sample the next token.
        sampled = self.sample(logits, sampling_metadata)
        # Convert sampled token ids to int64 (long) type to ensure compatibility
        # with subsequent operations that may use these values as indices.
        # This conversion is necessary because FlashInfer sampling operations
        # return int32 (while PyTorch argmax and topk return int64).
        sampled = sampled.long()

        # Gather the logprobs of the topk and sampled token (if requested).
        # Get logprobs and rank tensors (if requested)
        logprobs_tensors = None if num_logprobs is None else \
            self.gather_logprobs(raw_logprobs, num_logprobs, token_ids=sampled)

        # Use int32 to reduce the tensor size.
        sampled = sampled.to(torch.int32)

        # These are GPU tensors.
        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1], where each row represents one generated
            # token per request.
            sampled_token_ids=sampled.unsqueeze(-1),
            logprobs_tensors=logprobs_tensors,
        )
        return sampler_output

    def apply_temperature(
        self,
        logits: torch.Tensor,
        temp: torch.Tensor,
    ) -> torch.Tensor:
        # Use in-place division to avoid creating a new tensor.
        return logits.div_(temp.unsqueeze(dim=1))

    def greedy_sample(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1).view(-1)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        """Sample logits based on sampling metadata.

        The various logits processing functions called in this method
        may update the logits tensor in-place.
        """

        assert not (sampling_metadata.all_greedy
                    and sampling_metadata.all_random)
        if sampling_metadata.all_random:
            greedy_sampled = None
        else:
            greedy_sampled = self.greedy_sample(logits)
            if sampling_metadata.all_greedy:
                return greedy_sampled

        assert sampling_metadata.temperature is not None

        # Apply temperature.
        logits = self.apply_temperature(logits, sampling_metadata.temperature)

        # Apply logits processors that only apply to random sampling
        # (argmax invariant) # 
        for processor in sampling_metadata.logitsprocs.argmax_invariant:
            logits = processor.apply(logits)

        # Apply top_k and/or top_p.
        # random_sampled = self.topk_topp_sampler(
        #     logits,
        #     sampling_metadata.generators,
        #     sampling_metadata.top_k,
        #     sampling_metadata.top_p,
        # )   
        #

        # Check if longest word sampling is enabled
        longest_word_enable = get_param(sampling_metadata, "longest_word_enable", True, req_index=0)
        logger.info(f"Using longest_word_enable: {longest_word_enable}")

        # Debug logging
        if sampling_metadata.extra_args:
            logger.info(f"Legacy extra args: {sampling_metadata.extra_args}")
        if sampling_metadata.extra_args_per_request:
            logger.info(f"Per-request extra args: {sampling_metadata.extra_args_per_request}")

        filtered_logits = None
        if longest_word_enable:
            filtered_logits = apply_top_k_top_p(
                logits,
                sampling_metadata.top_k,
                sampling_metadata.top_p,
            )
            random_sampled = longest_word_sample(filtered_logits, self.token_lengths_gpu)
        else:
            # Use standard random sampling when longest_word_sample is disabled
            random_sampled = self.topk_topp_sampler(
                logits,
                sampling_metadata.generators,
                sampling_metadata.top_k,
                sampling_metadata.top_p,
            )

        # Check if forced EOS is enabled
        forced_eos_enable = get_param(sampling_metadata, "forced_eos_enable", True, req_index=0)
        logger.info(f"Using forced_eos_enable: {forced_eos_enable}")
        
        # For eos_position, we'll use per-request values in get_forced_eos_mask
        # Default eos_position from environment/initialization
        eos_position = self.eos_position


        if forced_eos_enable:
            logits_for_eos = filtered_logits if filtered_logits is not None else logits
            force_eos_mask = get_forced_eos_mask(
                sampling_metadata, eos_position, logits.device, logits_for_eos
            )
            if force_eos_mask is not None:
                eos_forced_count = force_eos_mask.sum().item()
                random_sampled = torch.where(force_eos_mask, self.eos_token_id, random_sampled)

        if greedy_sampled is None:
            return random_sampled

        sampled = torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
            out=greedy_sampled,  # Reuse tensor
        )

        return sampled


    def compute_logprobs(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.log_softmax(dim=-1, dtype=torch.float32)

    def gather_logprobs(
        self,
        logprobs: torch.Tensor,
        num_logprobs: int,
        token_ids: torch.Tensor,
    ) -> LogprobsTensors:
        """
        Gather logprobs for topk and sampled/prompt token.

        Args:
          logprobs: (num tokens) x (vocab) tensor
          num_logprobs: minimum number of logprobs to
                        retain per token
          token_ids: prompt tokens (if prompt logprobs)
                     or sampled tokens (if sampled
                     logprobs); 1D token ID tensor
                     with (num tokens) elements
                     Must be int64.

        Returns:
          Top-k int indices tensor, (num tokens) x (num_logprobs + 1)
          Top-k float logprobs tensor, (num tokens) x (num_logprobs + 1)
          Sampled token rank tensor, (num tokens)
        """
        assert token_ids.dtype == torch.int64
        # Find the topK values.
        topk_logprobs, topk_indices = torch.topk(logprobs,
                                                 num_logprobs,
                                                 dim=-1)

        # Get with the logprob of the prompt or sampled token.
        token_ids = token_ids.unsqueeze(-1)
        token_logprobs = logprobs.gather(-1, token_ids)

        # Compute the ranks of the actual token.
        token_ranks = (logprobs >= token_logprobs).sum(-1)

        # Concatenate together with the topk.
        indices = torch.cat((token_ids, topk_indices), dim=1)
        logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)

        # Use int32 to reduce the tensor size.
        indices = indices.to(torch.int32)

        return LogprobsTensors(indices, logprobs, token_ranks)

    def apply_penalties(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        if not sampling_metadata.no_penalties:
            assert sampling_metadata.prompt_token_ids is not None
            logits = apply_all_penalties(
                logits,
                sampling_metadata.prompt_token_ids,
                sampling_metadata.presence_penalties,
                sampling_metadata.frequency_penalties,
                sampling_metadata.repetition_penalties,
                sampling_metadata.output_token_ids,
            )
        return logits

    def apply_allowed_token_ids(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        if sampling_metadata.allowed_token_ids_mask is not None:
            logits.masked_fill_(sampling_metadata.allowed_token_ids_mask,
                                float("-inf"))
        return logits

    def apply_bad_words(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        if sampling_metadata.bad_words_token_ids:
            apply_bad_words(
                logits,
                sampling_metadata.bad_words_token_ids,
                sampling_metadata.output_token_ids,
            )
        return logits
