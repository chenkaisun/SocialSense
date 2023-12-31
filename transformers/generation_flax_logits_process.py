# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from abc import ABC

import jax
import jax.lax as lax
import jax.numpy as jnp
import jaxlib.xla_extension as jax_xla

from .file_utils import add_start_docstrings
from .utils.logging import get_logger


logger = get_logger(__name__)


LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`jax_xla.DeviceArray` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.PreTrainedTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        scores (:obj:`jax_xla.DeviceArray` of shape :obj:`(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search
        kwargs:
            Additional logits processor specific kwargs.

    Return:
        :obj:`jax_xla.DeviceArray` of shape :obj:`(batch_size, config.vocab_size)`: The processed prediction scores.

"""


class FlaxLogitsProcessor(ABC):
    """Abstract base class for all logit processors that can be applied during generation."""

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: jax_xla.DeviceArray, scores: jax_xla.DeviceArray) -> jax_xla.DeviceArray:
        """Flax method for processing logits."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class FlaxLogitsWarper(ABC):
    """Abstract base class for all logit warpers that can be applied during generation with multinomial sampling."""

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: jax_xla.DeviceArray, scores: jax_xla.DeviceArray) -> jax_xla.DeviceArray:
        """Flax method for warping logits."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class FlaxLogitsProcessorList(list):
    """
    This class can be used to create a list of :class:`~transformers.FlaxLogitsProcessor` or
    :class:`~transformers.FlaxLogitsWarper` to subsequently process a :obj:`scores` input tensor. This class inherits
    from list and adds a specific `__call__` method to apply each :class:`~transformers.FlaxLogitsProcessor` or
    :class:`~transformers.FlaxLogitsWarper` to the inputs.
    """

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: jax_xla.DeviceArray, scores: jax_xla.DeviceArray, cur_len: int, **kwargs
    ) -> jax_xla.DeviceArray:
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 3:
                assert all(
                    arg in kwargs for arg in list(function_args.keys())[2:]
                ), f"Make sure that all the required parameters: {list(function_args.keys())} for {processor.__class__} are passed to the logits processor."
                scores = processor(input_ids, scores, cur_len, **kwargs)
            else:
                scores = processor(input_ids, scores, cur_len)
        return scores


class FlaxTemperatureLogitsWarper(FlaxLogitsWarper):
    r"""
    :class:`transformers.LogitsWarper` for temperature (exponential scaling output probability distribution).

    Args:
        temperature (:obj:`float`):
            The value used to module the logits distribution.
    """

    def __init__(self, temperature: float):
        if not isinstance(temperature, float) or not (temperature > 0):
            raise ValueError(f"`temperature` has to be a strictly positive float, but is {temperature}")

        self.temperature = temperature

    def __call__(
        self, input_ids: jax_xla.DeviceArray, scores: jax_xla.DeviceArray, cur_len: int
    ) -> jax_xla.DeviceArray:
        scores = scores / self.temperature
        return scores


class FlaxTopPLogitsWarper(FlaxLogitsWarper):
    """
    :class:`transformers.LogitsWarper` that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <=
    prob_cut_off.

    Args:
        top_p (:obj:`float`):
            If set to < 1, only the most probable tokens with probabilities that add up to :obj:`top_p` or higher are
            kept for generation.
        filter_value (:obj:`float`, `optional`, defaults to :obj:`-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (:obj:`int`, `optional`, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if not isinstance(top_p, float) or (top_p < 0 or top_p > 1.0):
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(
        self, input_ids: jax_xla.DeviceArray, scores: jax_xla.DeviceArray, cur_len: int
    ) -> jax_xla.DeviceArray:
        topk_scores, topk_indices = lax.top_k(scores, scores.shape[-1])

        mask_scores = jnp.full_like(scores, self.filter_value)
        cumulative_probs = jax.nn.softmax(topk_scores, axis=-1).cumsum(axis=-1)
        score_mask = cumulative_probs < self.top_p

        # include the token that is higher than top_p as well
        score_mask |= jax.ops.index_update(jnp.roll(score_mask, 1), jax.ops.index[:, 0], True)

        # min tokens to keep
        score_mask = jax.ops.index_update(score_mask, jax.ops.index[:, : self.min_tokens_to_keep], True)

        topk_next_scores = jnp.where(score_mask, topk_scores, mask_scores)
        next_scores = jax.lax.sort_key_val(topk_indices, topk_next_scores)[-1]

        return next_scores


class FlaxTopKLogitsWarper(FlaxLogitsWarper):
    r"""
    :class:`transformers.LogitsWarper` that performs top-k, i.e. restricting to the k highest probability elements.

    Args:
        top_k (:obj:`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (:obj:`float`, `optional`, defaults to :obj:`-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (:obj:`int`, `optional`, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_k: int, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")

        self.top_k = top_k
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(
        self, input_ids: jax_xla.DeviceArray, scores: jax_xla.DeviceArray, cur_len: int
    ) -> jax_xla.DeviceArray:
        batch_size, vocab_size = scores.shape
        next_scores_flat = jnp.full(batch_size * vocab_size, self.filter_value)

        topk = min(max(self.top_k, self.min_tokens_to_keep), scores.shape[-1])  # Safety check
        topk_scores, topk_indices = lax.top_k(scores, topk)
        shift = jnp.broadcast_to((jnp.arange(batch_size) * vocab_size)[:, None], (batch_size, topk)).flatten()
        topk_scores_flat = topk_scores.flatten()
        topk_indices_flat = topk_indices.flatten() + shift

        next_scores_flat = jax.ops.index_update(next_scores_flat, topk_indices_flat, topk_scores_flat)
        next_scores = next_scores_flat.reshape(batch_size, vocab_size)
        return next_scores


class FlaxForcedBOSTokenLogitsProcessor(FlaxLogitsProcessor):
    r"""
    :class:`~transformers.FlaxLogitsProcessor` that enforces the specified token as the first generated token.

    Args:
        bos_token_id (:obj:`int`):
            The id of the token to force as the first generated token.
    """

    def __init__(self, bos_token_id: int):
        self.bos_token_id = bos_token_id

    def __call__(
        self, input_ids: jax_xla.DeviceArray, scores: jax_xla.DeviceArray, cur_len: int
    ) -> jax_xla.DeviceArray:
        new_scores = jnp.full(scores.shape, -float("inf"))

        apply_penalty = 1 - jnp.bool_(cur_len - 1)

        scores = jnp.where(
            apply_penalty, jax.ops.index_update(new_scores, jax.ops.index[:, self.bos_token_id], 0), scores
        )

        return scores


class FlaxForcedEOSTokenLogitsProcessor(FlaxLogitsProcessor):
    r"""
    :class:`~transformers.FlaxLogitsProcessor` that enforces the specified token as the last generated token when
    :obj:`max_length` is reached.

    Args:
        max_length (:obj:`int`):
            The maximum length of the sequence to be generated.
        eos_token_id (:obj:`int`):
            The id of the token to force as the last generated token when :obj:`max_length` is reached.
    """

    def __init__(self, max_length: int, eos_token_id: int):
        self.max_length = max_length
        self.eos_token_id = eos_token_id

    def __call__(
        self, input_ids: jax_xla.DeviceArray, scores: jax_xla.DeviceArray, cur_len: int
    ) -> jax_xla.DeviceArray:
        new_scores = jnp.full(scores.shape, -float("inf"))

        apply_penalty = 1 - jnp.bool_(cur_len - self.max_length + 1)

        scores = jnp.where(
            apply_penalty, jax.ops.index_update(new_scores, jax.ops.index[:, self.eos_token_id], 0), scores
        )

        return scores


class FlaxMinLengthLogitsProcessor(FlaxLogitsProcessor):
    r"""
    :class:`transformers.FlaxLogitsProcessor` enforcing a min-length by setting EOS probability to 0.

    Args:
        min_length (:obj:`int`):
            The minimum length below which the score of :obj:`eos_token_id` is set to :obj:`-float("Inf")`.
        eos_token_id (:obj:`int`):
            The id of the `end-of-sequence` token.
    """

    def __init__(self, min_length: int, eos_token_id: int):
        if not isinstance(min_length, int) or min_length < 0:
            raise ValueError(f"`min_length` has to be a positive integer, but is {min_length}")

        if not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise ValueError(f"`eos_token_id` has to be a positive integer, but is {eos_token_id}")

        self.min_length = min_length
        self.eos_token_id = eos_token_id

    def __call__(
        self, input_ids: jax_xla.DeviceArray, scores: jax_xla.DeviceArray, cur_len: int
    ) -> jax_xla.DeviceArray:

        # create boolean flag to decide if min length penalty should be applied
        apply_penalty = 1 - jnp.clip(cur_len - self.min_length, 0, 1)

        scores = jnp.where(
            apply_penalty, jax.ops.index_update(scores, jax.ops.index[:, self.eos_token_id], -float("inf")), scores
        )

        return scores
