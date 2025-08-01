# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import time
from collections.abc import Sequence
from typing import Any, Optional, Union

import msgspec
import torch

from vllm.lora.request import LoRARequest
from vllm.multimodal import MultiModalKwargs
from vllm.multimodal.inputs import PlaceholderRange
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import LogprobsLists, LogprobsTensors

# These are possible values of RequestOutput.finish_reason,
# so form part of the external API.
FINISH_REASON_STRINGS = ("stop", "length", "abort")


class FinishReason(enum.IntEnum):
    """
    Reason a request finished - stop, length, or abort.

    Int rather than Str for more compact serialization.

    stop - a stop string was emitted
    length - max_tokens was consumed, or max_model_len was reached
    abort - aborted for another reason

    """
    STOP = 0
    LENGTH = 1
    ABORT = 2

    def __str__(self):
        return FINISH_REASON_STRINGS[self.value]


class EngineCoreRequest(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True,  # type: ignore[call-arg]
        gc=False):  # type: ignore[call-arg]

    request_id: str
    prompt_token_ids: list[int]
    mm_inputs: Optional[Sequence[Optional[MultiModalKwargs]]]
    mm_hashes: Optional[list[str]]
    mm_placeholders: Optional[list[PlaceholderRange]]
    sampling_params: Optional[SamplingParams]
    pooling_params: Optional[PoolingParams]
    eos_token_id: Optional[int]
    arrival_time: float
    lora_request: Optional[LoRARequest]
    cache_salt: Optional[str]
    data_parallel_rank: Optional[int]

    # Index of the client, used to ensure outputs are sent back to the same
    # client for this request when scaling out the front-end.
    client_index: int = 0

    # Used in DP case to indicate which wave of requests this is expected to
    # belong to, to cover a race condition where the request is sent before
    # a wave finished notification is received.
    current_wave: int = 0
    priority: int = 0


class EngineCoreEventType(enum.IntEnum):
    """The type of engine core request event."""
    QUEUED = 1
    SCHEDULED = 2
    PREEMPTED = 3


class EngineCoreEvent(msgspec.Struct):
    """A timestamped engine core event associated with a request.

    The timestamp is a monotonic timestamps and is used for by the engine
    frontend to calculate intervals between engine core events. These
    timestamps should not be compared with timestamps from other processes.
    """
    type: EngineCoreEventType
    timestamp: float

    @classmethod
    def new_event(cls,
                  event_type: EngineCoreEventType,
                  timestamp: Optional[float] = None) -> "EngineCoreEvent":
        timestamp = time.monotonic() if timestamp is None else timestamp
        return cls(event_type, timestamp)


class EngineCoreOutput(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True,  # type: ignore[call-arg]
        gc=False):  # type: ignore[call-arg]

    request_id: str
    new_token_ids: list[int]

    new_logprobs: Optional[LogprobsLists] = None
    new_prompt_logprobs_tensors: Optional[LogprobsTensors] = None

    pooling_output: Optional[torch.Tensor] = None

    finish_reason: Optional[FinishReason] = None
    stop_reason: Union[int, str, None] = None
    events: Optional[list[EngineCoreEvent]] = None
    kv_transfer_params: Optional[dict[str, Any]] = None

    # The number of tokens with prefix cache hits.
    num_cached_tokens: int = 0

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None


class UtilityOutput(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        gc=False):  # type: ignore[call-arg]

    call_id: int

    # Non-None implies the call failed, result should be None.
    failure_message: Optional[str] = None
    result: Any = None


class EngineCoreOutputs(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True,  # type: ignore[call-arg]
        gc=False):  # type: ignore[call-arg]

    #NOTE(Nick): We could consider ways to make this more compact,
    # e.g. columnwise layout

    engine_index: int = 0

    # [num_reqs]
    outputs: list[EngineCoreOutput] = []
    scheduler_stats: Optional[SchedulerStats] = None
    timestamp: float = 0.0

    utility_output: Optional[UtilityOutput] = None
    finished_requests: Optional[set[str]] = None

    # In DP case, used to signal that the current wave of requests
    # has finished and the engines are paused.
    wave_complete: Optional[int] = None
    # In DP case, used to signal that a request was received for an
    # "old" wave, so the next wave needs to be started in other engines.
    start_wave: Optional[int] = None

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.monotonic()


class EngineCoreRequestType(enum.Enum):
    """
    Request types defined as hex byte strings, so it can be sent over sockets
    without separate encoding step.
    """
    ADD = b'\x00'
    ABORT = b'\x01'
    START_DP_WAVE = b'\x02'
    UTILITY = b'\x03'
    # Sentinel used within EngineCoreProc.
    EXECUTOR_FAILED = b'\x04'


class ReconfigureDistributedRequest(msgspec.Struct):
    new_data_parallel_size: int
    new_data_parallel_rank: int
    new_data_parallel_rank_local: int
    new_data_parallel_master_ip: str
    new_data_parallel_master_port: int


class ReconfigureRankType(enum.IntEnum):
    """
    Rank type for reconfiguring distributed request.
    """
    KEEP_CURRENT_RANK = -1
    SHUTDOWN_CURRENT_RANK = -2
