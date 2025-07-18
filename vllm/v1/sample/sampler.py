# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A layer that samples the next tokens from the model's outputs."""

import torch
import torch.nn as nn

import vllm.envs as envs
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

# Probability ratio threshold for EOS token logic
# When EOS is in top-k, other tokens must have probability >= eos_prob/EOS_PROB_RATIO
# This prevents selecting extremely unlikely tokens when EOS is available
# (e.g., prevents cases where chosen_token_prob is 100x lower than eos_prob)
EOS_PROB_RATIO = 100.0
MIN_PROB_THRESHOLD = 0.001  # Minimum probability threshold for valid tokens


def longest_word_sample(
    logits: torch.Tensor,
    token_lengths: torch.Tensor,
    top_k: int = 10,
    mix_ratio: float = 0.5,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """
    Sample tokens by mixing longest word selection with probability-based sampling.

    If eos_token_id is provided and present in top-k tokens, applies a probability
    threshold rule: only tokens with probability >= eos_prob/EOS_PROB_RATIO can be selected.
    If no tokens meet this threshold, EOS is forced. This prevents selecting tokens
    that are significantly less likely than EOS.

    Args:
        logits: Tensor of shape [batch_size, vocab_size]
        token_lengths: Tensor of shape [vocab_size] with character lengths
        top_k: Number of top candidates to consider
        mix_ratio: Balance between longest-word (1.0) and probability-based (0.0)
        eos_token_id: If provided and present in top-k, enables threshold logic

    Returns:
        Tensor of shape [batch_size] - selected token indices
    """
    k = min(top_k, logits.size(-1))
    topk_logits, topk_idx = torch.topk(logits, k, dim=-1)  # [B, k]

    # 2. Calculate normalized probability scores for top-k tokens
    prob_score = F.softmax(topk_logits, dim=-1)  # [B, k]
    length_score = F.softmax(
        token_lengths[topk_idx.clamp_max(token_lengths.size(0) - 1)].float(),
        dim=-1,
    )

    # 3. Mix probability and length scores based on mix_ratio
    mix_score = (1.0 - mix_ratio) * prob_score + mix_ratio * length_score

    # Apply minimum probability threshold of 0.005
    prob_threshold = torch.tensor(MIN_PROB_THRESHOLD, device=prob_score.device)
    valid_mask = prob_score >= prob_threshold
    mix_score = torch.where(valid_mask, mix_score, -torch.inf)

    # 4. Apply EOS probability threshold logic if EOS token is specified
    if eos_token_id is not None:
        # Identify which batches have EOS token in their top-k candidates
        eos_in_topk = (topk_idx == eos_token_id) & torch.isfinite(
            topk_logits
        )  # [B, k]
        rows_with_eos = eos_in_topk.any(dim=-1)  # [B]

        # Apply threshold logic only if at least one batch contains EOS
        if rows_with_eos.any():
            # Extract EOS probability for each batch (0 if EOS not present)
            eos_prob = (prob_score * eos_in_topk).sum(
                dim=-1, keepdim=True
            )  # [B, 1]

            # Calculate threshold: tokens must have prob >= eos_prob/EOS_PROB_RATIO
            # but also enforce minimum threshold of 0.005
            eos_threshold = torch.maximum(
                eos_prob / EOS_PROB_RATIO,
                torch.tensor(0.005, device=eos_prob.device)
            )  # [B, 1]

            # Mark tokens that meet the EOS probability threshold
            valid_candidate_mask = prob_score >= eos_threshold  # [B, k]

            # Further mask out tokens based on EOS threshold
            # Only apply additional masking for rows that have EOS
            mix_score = torch.where(
                valid_candidate_mask | ~rows_with_eos.unsqueeze(-1),
                mix_score,
                -torch.inf,
            )
    # 5. Select token with highest (possibly masked) mix_score
    best_in_topk = mix_score.argmax(dim=-1, keepdim=True)  # [B, 1]
    chosen = topk_idx.gather(-1, best_in_topk).squeeze(-1)  # [B]



    return chosen


_SAMPLING_EPS = 1e-5


logger = init_logger(__name__)
# docker logs -n 5000 orchestrator 2>&1 | grep  -A 10 -B 10 -Ei "❌"


class Sampler(nn.Module):

    def __init__(self, token_lengths_gpu, eos_token_id: int | None = None):
        super().__init__()
        self.topk_topp_sampler = TopKTopPSampler()
        self.pin_memory = is_pin_memory_available()
        self.token_lengths_gpu = token_lengths_gpu
        self.mix_ratio = envs.VLLM_MIX_RATIO
        if envs.VLLM_EOS_TOKEN_USAGE:
            self.eos_token_id = eos_token_id
        else:
            logger.info("EOS token usage is disabled, setting eos_token_id to None.")
            self.eos_token_id = None
        logger.info(f"Sampler initialized with mix_ratio={self.mix_ratio} "
                    f"(0.0=pure probability, 1.0=pure longest word), {eos_token_id=}")

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
        # (argmax invariant)
        for processor in sampling_metadata.logitsprocs.argmax_invariant:
            logits = processor.apply(logits)

        # Apply top_k and/or top_p.
        # random_sampled = self.topk_topp_sampler(
        #     logits,
        #     sampling_metadata.generators,
        #     sampling_metadata.top_k,
        #     sampling_metadata.top_p,
        # )
        filtered_logits = apply_top_k_top_p(
            logits,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )

        random_sampled = longest_word_sample(
            filtered_logits, 
            self.token_lengths_gpu, 
            top_k=sampling_metadata.top_k.max().item() if sampling_metadata.top_k is not None else 10,
            mix_ratio=self.mix_ratio,
            eos_token_id=self.eos_token_id,
        )

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
