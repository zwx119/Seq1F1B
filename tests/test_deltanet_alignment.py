#!/usr/bin/env python
# Copyright (c) 2024, Seq1F1B-DeltaNet Contributors.
"""
Alignment test: verify DeltaNet Seq1F1B splitting produces identical
hidden states compared to serial (full-sequence) execution.

This test uses the REAL Megatron PP pipeline with real DeltaNetTransformerLayer.

Launch (4 GPU, PP=4):
  # No-SP baseline (pipe_sp_splits=1):
  GPUS_PER_NODE=4 PP_SP=1 bash tests/run_alignment_test.sh

  # Seq1F1B (pipe_sp_splits=4):
  GPUS_PER_NODE=4 PP_SP=4 bash tests/run_alignment_test.sh

Or use the all-in-one script:
  bash tests/run_alignment_compare.sh
"""

import os
import sys
import torch
import torch.distributed as dist

from megatron import get_args, print_rank_0
from megatron.core import parallel_state, tensor_parallel
from megatron.core.enums import ModelType
from megatron.training import pretrain
from megatron.model import GPTModel
from megatron.arguments import core_transformer_config_from_args
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.core.pipeline_parallel.sp_utils import get_splits

# ── Global state for capturing hidden states ──
_captured_outputs = []
_micro_sp_offset = -1
_global_input = None

def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    print_rank_0('building GPT model (alignment test) ...')
    config = core_transformer_config_from_args(get_args())
    model = GPTModel(
        config,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model


def get_batch(data_iterator):
    """Generate a batch from the data iterator (same as pretrain_gpt.py)."""
    args = get_args()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and position ids.
    from megatron.utils import get_ltor_masks_and_position_ids
    from megatron import get_tokenizer
    tokenizer = get_tokenizer()
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask, position_ids


def get_batch_sp_func_factory():
    """Creates the SP data splitter, same logic as pretrain_gpt.py."""
    offset = -1
    global_data = None

    def get_data(*args, **kwargs):
        nonlocal global_data, offset
        pipe_sp = get_args().pipe_sp_splits
        if offset == -1 or offset + 1 == pipe_sp:
            global_data = get_batch(*args, **kwargs)
        offset = (offset + 1) % pipe_sp
        tokens, labels, loss_mask, attention_mask, position_ids = global_data
        seq_length = tokens.size(1)
        # average strategy
        tokens = tokens.chunk(pipe_sp, dim=1)[offset]
        labels = labels.chunk(pipe_sp, dim=1)[offset]
        loss_mask._start = seq_length // pipe_sp * offset
        loss_mask._end = seq_length // pipe_sp * (offset + 1)
        position_ids = position_ids.chunk(pipe_sp, dim=1)[offset]
        return tokens, labels, loss_mask, attention_mask, position_ids, offset

    return get_data


_get_batch_sp = get_batch_sp_func_factory()


def loss_func(loss_mask, output_tensor):
    """Loss function that also captures the output for comparison."""
    losses = output_tensor.float()
    start = loss_mask._start
    end = loss_mask._end
    loss_mask_p = loss_mask[:, start:end]
    loss_mask_full = loss_mask.contiguous().view(-1).float()
    args = get_args()
    if args.pipe_sp_splits > 1:
        loss_mask_p = loss_mask_p.contiguous().view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask_p) / loss_mask_full.sum() * args.pipe_sp_splits
    else:
        loss = torch.sum(losses.view(-1) * loss_mask_full) / loss_mask_full.sum()

    # Capture loss value on last PP stage
    if parallel_state.is_pipeline_last_stage():
        _captured_outputs.append(loss.detach().clone())

    from megatron.utils import average_losses_across_data_parallel_group
    averaged_loss = average_losses_across_data_parallel_group([loss])
    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step with SP data splitting."""
    args = get_args()
    from functools import partial
    tokens, labels, loss_mask, attention_mask, position_ids, offset = _get_batch_sp(data_iterator)
    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels, micro_sp_idx=offset)
    return output_tensor, partial(loss_func, loss_mask)


class SyntheticDataset(torch.utils.data.Dataset):
    """Dummy dataset that returns random token sequences."""
    def __init__(self, num_samples, seq_length, vocab_size, seed=42):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Deterministic per-sample random data
        rng = torch.Generator()
        rng.manual_seed(self.seed + idx)
        tokens = torch.randint(0, self.vocab_size, (self.seq_length + 1,), generator=rng)
        return {'text': tokens}


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Return synthetic datasets so training loop runs."""
    args = get_args()
    train_ds = SyntheticDataset(
        num_samples=train_val_test_num_samples[0],
        seq_length=args.seq_length,
        vocab_size=args.padded_vocab_size,
        seed=args.seed,
    )
    return train_ds, None, None


if __name__ == "__main__":
    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
