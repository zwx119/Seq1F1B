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
import argparse

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
_SAVE_SECOND_ITER = int(os.environ.get("SAVE_SECOND_ITER", "10"))  # also save at this iter
_SAVE_LAST_N = int(os.environ.get("SAVE_LAST_N", "3"))           # save last N iters
# For layer-level stats dumping
_pre_fwd_count = 0
_current_fwd_meta = {}  # populated by encoder pre-hook: {'cur_train_iter':int, 'micro_idx':int, 'chunk_idx':int}
layer_forward_stats = {}  # iter -> layer_idx -> list of stats dicts
layer_grad_stats = {}     # iter -> layer_idx -> list of grad norm floats


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
        """Capture encoder output at specific iterations only.

        Uses metadata populated by the encoder pre-forward hook so that
        per-layer hooks can rely on the same iteration/microbatch values.
        """
        args = get_args()
        sp = args.pipe_sp_splits
        num_micro = args.global_batch_size // args.micro_batch_size
        fwd_per_iter = sp * num_micro  # total forward calls per training iteration

        # If the pre-hook populated metadata, use that; otherwise, fall back
        # to computing based on a local counter.
        meta = _current_fwd_meta.get('meta') if isinstance(_current_fwd_meta.get('meta'), dict) else None
        if meta is not None:
            cur_train_iter = meta['cur_train_iter']
            micro_idx = meta['micro_idx']
            chunk_idx = meta['chunk_idx']
        else:
            # Fallback: increment shared counter (older behavior)
            global _fwd_count
            _fwd_count += 1
            cur_train_iter = (_fwd_count - 1) // fwd_per_iter + 1
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
            keep_iters = {_SAVE_EARLY_ITER, _SAVE_SECOND_ITER}
            for j in range(max(1, cur_train_iter - _SAVE_LAST_N + 1), cur_train_iter + 1):
                keep_iters.add(j)
            for k in list(_hs_chunks.keys()):
                if k not in keep_iters:
                    del _hs_chunks[k]

    encoder.register_forward_hook(_capture_hook)
    # Pre-forward hook on encoder to populate iteration metadata before
    # layer-level forward hooks run. This lets layer hooks know which
    # training iteration and microbatch they belong to.
    def _pre_hook(module, input):
        global _pre_fwd_count
        args = get_args()
        sp = args.pipe_sp_splits
        num_micro = args.global_batch_size // args.micro_batch_size
        fwd_per_iter = sp * num_micro

        _pre_fwd_count += 1
        cur_train_iter = (_pre_fwd_count - 1) // fwd_per_iter + 1
        within_iter = (_pre_fwd_count - 1) % fwd_per_iter
        micro_idx = within_iter // sp
        chunk_idx = within_iter % sp
        _current_fwd_meta['meta'] = {
            'cur_train_iter': cur_train_iter,
            'micro_idx': micro_idx,
            'chunk_idx': chunk_idx,
        }

    encoder.register_forward_pre_hook(_pre_hook)

    # Optionally register per-layer hooks to dump forward and grad stats.
    args = get_args()
    if getattr(args, 'dump_layer_stats', False):
        # For each transformer layer, register a forward hook that records
        # simple statistics (norm, mean, max) of the layer output when
        # the microbatch is the first microbatch of an iteration, and
        # only for the early and second iteration we care about.
        for li, layer in enumerate(getattr(encoder, 'layers', [])):
            def make_layer_hook(layer_idx):
                def _layer_hook(module, input, output):
                    meta = _current_fwd_meta.get('meta')
                    if meta is None:
                        return
                    cur_train_iter = meta['cur_train_iter']
                    micro_idx = meta['micro_idx']
                    # Only capture the first microbatch of each iter
                    if micro_idx != 0:
                        return
                    # Only capture the two iters we care about
                    if cur_train_iter not in (_SAVE_EARLY_ITER, _SAVE_SECOND_ITER):
                        return

                    # compute stats
                    out = output.detach()
                    stats = {
                        'norm': out.float().norm().item(),
                        'mean': out.float().mean().item(),
                        'maxabs': out.float().abs().max().item(),
                    }
                    layer_forward_stats.setdefault(cur_train_iter, {}).setdefault(layer_idx, []).append(stats)

                    # register a backward hook to capture grad norms for this
                    # output tensor when backward runs
                    def _save_grad(grad):
                        try:
                            gnorm = grad.detach().float().norm().item()
                        except Exception:
                            gnorm = None
                        layer_grad_stats.setdefault(cur_train_iter, {}).setdefault(layer_idx, []).append(gnorm)

                    if isinstance(output, torch.Tensor) and output.requires_grad:
                        try:
                            output.register_hook(_save_grad)
                        except Exception:
                            pass
                return _layer_hook
            layer.register_forward_hook(make_layer_hook(li))

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


# Whether to disable state passing between SP chunks (ablation)
_DISABLE_STATE_PASSING = os.environ.get("DISABLE_STATE_PASSING", "0") == "1"


def forward_step(data_iterator, model):
    """Forward step with SP data splitting."""
    from functools import partial
    tokens, labels, loss_mask, attention_mask, position_ids, offset = _get_batch_sp(data_iterator)
    # Ablation: force micro_sp_idx=0 for every chunk → no state passing
    effective_offset = 0 if _DISABLE_STATE_PASSING else offset
    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels, micro_sp_idx=effective_offset)
    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build real train/valid/test datasets (same as pretrain_gpt.py)."""
    from megatron.data.gpt_dataset import build_train_valid_test_datasets
    from megatron import print_rank_0
    args = get_args()
    print_rank_0('> building train, validation, and test datasets for alignment test ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        train_data_prefix=args.train_data_path,
        valid_data_prefix=args.valid_data_path,
        test_data_prefix=args.test_data_path,
        data_cache_path=args.data_cache_path)
    print_rank_0("> finished creating alignment test datasets ...")
    return train_ds, valid_ds, test_ds


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

    tag = "nostate" if _DISABLE_STATE_PASSING else f"sp{sp}"
    save_path = os.path.join(save_dir, f"hs_{tag}_stage{rank}.pt")
    torch.save(hs_dict, save_path)
    saved_iters = sorted(hs_dict.keys())
    print(f"[Stage {rank}] Saved iters {saved_iters} "
          f"({tag}) to {save_path}")

    dist.barrier()
    # Also save per-layer forward/grad stats if collected
    if layer_forward_stats or layer_grad_stats:
        stats_path = os.path.join(save_dir, f"layer_stats_{tag}_stage{rank}.pt")
        torch.save({'forward': layer_forward_stats, 'grad': layer_grad_stats}, stats_path)
        print(f"[Stage {rank}] Saved layer stats to {stats_path}")

    # If we have both early and second iter hidden states, print a small diff summary
    try:
        if _SAVE_EARLY_ITER in hs_dict and _SAVE_SECOND_ITER in hs_dict:
            a = hs_dict[_SAVE_EARLY_ITER]
            b = hs_dict[_SAVE_SECOND_ITER]
            # compute simple norms of difference
            diff = (a - b).float()
            print(f"[Stage {rank}] Early vs Second iter hidden states diff: norm={diff.norm().item():.6f}, mean_abs={diff.abs().mean().item():.6e}")
    except Exception:
        pass


# ── Monkey-patch pretrain to save hidden states after training ──
_original_pretrain = pretrain

def _patched_pretrain(*args, **kwargs):
    _original_pretrain(*args, **kwargs)
    _save_hidden_states()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp32', action='store_true', help='Use fp32 precision')
    parser.add_argument('--dump-layer-stats', action='store_true', help='Dump per-layer forward and grad stats for first two iters')
    # Parse known args to avoid interfering with Megatron's own parser
    args, unknown = parser.parse_known_args()
    # Ensure megatron's global parser knows about --fp32 too by providing
    # an extra_args_provider that registers the flag. This avoids
    # argparse failures when Megatron's parse_args() is called inside
    # initialize_megatron (which runs in each spawned process).
    def _extra_args_provider(parser):
        parser.add_argument('--fp32', action='store_true', help='Use fp32 precision')
        parser.add_argument('--dump-layer-stats', action='store_true', help='Dump per-layer forward and grad stats for first two iters')
        return parser

    # Optionally, you can set a global flag or patch get_args() if needed
    # If the user requested --fp32, pass defaults to Megatron to disable
    # mixed precision and force fp32 parameters/computations.
    args_defaults = {'tokenizer_type': 'GPT2BPETokenizer'}
    if args.fp32:
        # Ensure spawned processes also run in fp32 by setting defaults
        # for fp16/bf16 to False. Megatron's argument parsing will respect
        # these defaults when flags are not explicitly set.
        args_defaults.update({'fp16': False, 'bf16': False})

    _patched_pretrain(train_valid_test_datasets_provider,
                      model_provider,
                      ModelType.encoder_or_decoder,
                      forward_step,
                      extra_args_provider=_extra_args_provider,
                      args_defaults=args_defaults)
