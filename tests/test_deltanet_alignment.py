#!/usr/bin/env python
# Copyright (c) 2024, Seq1F1B-DeltaNet Contributors.
"""
Alignment test: verify DeltaNet Seq1F1B splitting produces identical
hidden states compared to serial (full-sequence) execution.

This test uses the REAL Megatron PP pipeline with real DeltaNetTransformerLayer.
It captures hidden states (ParallelTransformer output) on EVERY PP stage
and saves them to disk for comparison between SP=1 and SP=4.

Launch:
  DATA_PATH=/path/to/data bash tests/run_alignment_compare.sh
"""

import os
import sys
import torch
import torch.distributed as dist
from collections import deque

from megatron import get_args, print_rank_0
from megatron.core import parallel_state, tensor_parallel
from megatron.core.enums import ModelType
from megatron.training import pretrain
from megatron.model import GPTModel
from megatron.arguments import core_transformer_config_from_args
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.core.pipeline_parallel.sp_utils import get_splits

# ── Global state for capturing hidden states ──
_hs_chunks = {}    # dict: iter_number -> list of chunks for that iter
_fwd_count = 0     # total forward calls on this stage
_SAVE_EARLY_ITER = int(os.environ.get("SAVE_EARLY_ITER", "1"))  # save at this iter (1 = before any weight update)
_SAVE_LAST_N = int(os.environ.get("SAVE_LAST_N", "0"))           # save last N iters (0 = skip for long training)


def _get_save_dir():
    args = get_args()
    save_dir = os.path.join(os.path.dirname(__file__), "alignment_outputs")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def model_provider(pre_process=True, post_process=True):
    """Build the model and register hidden-state capture hook."""
    print_rank_0('building GPT model (alignment test) ...')
    config = core_transformer_config_from_args(get_args())
    model = GPTModel(
        config,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )

    # Register forward hook on the encoder (ParallelTransformer) to
    # capture hidden states on this PP stage.
    encoder = model.language_model.encoder

    def _capture_hook(module, input, output):
        """Capture encoder output at specific iterations only."""
        global _fwd_count
        args = get_args()
        sp = args.pipe_sp_splits
        num_micro = args.global_batch_size // args.micro_batch_size
        fwd_per_iter = sp * num_micro  # total forward calls per training iteration

        _fwd_count += 1
        # Which training iteration is this? (1-indexed)
        cur_train_iter = (_fwd_count - 1) // fwd_per_iter + 1
        # Which microbatch within this iteration?
        within_iter = (_fwd_count - 1) % fwd_per_iter
        micro_idx = within_iter // sp
        chunk_idx = within_iter % sp

        # Only capture the FIRST microbatch of each iteration (micro_idx == 0)
        if micro_idx != 0:
            return

        # Buffer current iter; cleanup happens at iter boundary
        if cur_train_iter not in _hs_chunks:
            _hs_chunks[cur_train_iter] = []
        _hs_chunks[cur_train_iter].append(output.detach().clone())

        # At the end of first microbatch, prune old iters
        if chunk_idx == sp - 1:
            keep_iters = {_SAVE_EARLY_ITER}
            for j in range(max(1, cur_train_iter - _SAVE_LAST_N + 1), cur_train_iter + 1):
                keep_iters.add(j)
            for k in list(_hs_chunks.keys()):
                if k not in keep_iters:
                    del _hs_chunks[k]

    encoder.register_forward_hook(_capture_hook)

    return model


def get_batch(data_iterator):
    """Generate a batch from the data iterator (same as pretrain_gpt.py)."""
    args = get_args()
    keys = ['text']
    datatype = torch.int64
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()
    from megatron import get_tokenizer
    tokenizer = get_tokenizer()
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens, tokenizer.eod,
        args.reset_position_ids, args.reset_attention_mask, args.eod_mask_loss)
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
        tokens = tokens.chunk(pipe_sp, dim=1)[offset]
        labels = labels.chunk(pipe_sp, dim=1)[offset]
        loss_mask._start = seq_length // pipe_sp * offset
        loss_mask._end = seq_length // pipe_sp * (offset + 1)
        position_ids = position_ids.chunk(pipe_sp, dim=1)[offset]
        return tokens, labels, loss_mask, attention_mask, position_ids, offset

    return get_data

_get_batch_sp = get_batch_sp_func_factory()


def loss_func(loss_mask, output_tensor):
    """Loss function."""
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

    from megatron.utils import average_losses_across_data_parallel_group
    averaged_loss = average_losses_across_data_parallel_group([loss])
    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step with SP data splitting."""
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


def _save_hidden_states():
    """Called after training to save captured hidden states to disk."""
    args = get_args()
    rank = parallel_state.get_pipeline_model_parallel_rank()
    sp = args.pipe_sp_splits
    save_dir = _get_save_dir()

    # _hs_chunks is dict: iter_number -> list of sp chunks
    # Combine chunks per iter into full-seq tensors
    hs_dict = {}
    for iter_num, chunks in sorted(_hs_chunks.items()):
        if len(chunks) == sp:
            combined = torch.cat(chunks, dim=0)  # [s_full, b, h]
            hs_dict[iter_num] = combined

    save_path = os.path.join(save_dir, f"hs_sp{sp}_stage{rank}.pt")
    torch.save(hs_dict, save_path)
    saved_iters = sorted(hs_dict.keys())
    print(f"[Stage {rank}] Saved iters {saved_iters} "
          f"(sp={sp}) to {save_path}")

    dist.barrier()


# ── Monkey-patch pretrain to save hidden states after training ──
_original_pretrain = pretrain

def _patched_pretrain(*args, **kwargs):
    _original_pretrain(*args, **kwargs)
    _save_hidden_states()

if __name__ == "__main__":
    _patched_pretrain(train_valid_test_datasets_provider,
                      model_provider,
                      ModelType.encoder_or_decoder,
                      forward_step,
                      args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
