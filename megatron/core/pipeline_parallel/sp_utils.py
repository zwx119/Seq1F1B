from dataclasses import dataclass
import math
from megatron.core.pipeline_parallel.split_solver import solver
from megatron import get_args
import torch
@dataclass
class SeqTFlops:
    num_layers: int
    hidden_size: int
    ffn_size: int
    num_heads: int
    dim_head: int
    vocab_size: int
    softmax_layers: int = None
    deltanet_layers: int = 0
    deltanet_head_dim: int = None
    deltanet_chunk_size: int = 64
    deltanet_conv_size: int = 4
    deltanet_use_beta: bool = True
    deltanet_use_short_conv: bool = True
    deltanet_use_output_gate: bool = False

    def get_quadratic_layers(self):
        return self.num_layers if self.softmax_layers is None else self.softmax_layers

    def get_ffn_tflops(self, seqlen):
        ffn_tflops = 4 * seqlen * self.hidden_size * self.ffn_size
        return ffn_tflops

    def get_deltanet_core_tflops(self, seqlen):
        """Approximate per-DeltaNet-layer linear recurrent-core FLOPs.

        The shared q/k/v/o projections and MLP are counted in the common linear
        terms. This term accounts for DeltaNet-specific work that is linear in
        sequence length: KKT/WY preparation, state passing/output, short conv,
        beta projection, and the optional output gate projection.
        """
        if self.deltanet_layers <= 0:
            return 0

        head_dim = self.deltanet_head_dim or self.dim_head
        key_dim = head_dim
        value_dim = head_dim
        block = self.deltanet_chunk_size

        # K @ K^T over 64-token DeltaNet blocks.
        kkt = 2 * seqlen * block * self.num_heads * key_dim
        # Triangular solve/recompute applications to K and V representations.
        wy = 2 * seqlen * block * self.num_heads * (key_dim + value_dim)
        # State update plus output contraction, both linear in sequence length.
        state_and_output = 4 * seqlen * self.num_heads * key_dim * value_dim

        short_conv = 0
        if self.deltanet_use_short_conv:
            short_conv = (
                2 * seqlen * self.deltanet_conv_size * self.num_heads
                * (2 * key_dim + value_dim)
            )

        beta = 0
        if self.deltanet_use_beta:
            beta = 2 * seqlen * self.hidden_size * self.num_heads

        gate = 0
        if self.deltanet_use_output_gate:
            # The common projection term counts q/k/v/o. Output gating adds one
            # extra hidden->hidden projection for g.
            gate = 2 * seqlen * self.hidden_size * self.hidden_size

        return kkt + wy + state_and_output + short_conv + beta + gate

    def get_emb_tflops(self, seqlen):
        embed_tflops = 2 * seqlen * self.hidden_size * self.vocab_size
        embed_proj_tflops = 2 * seqlen * self.hidden_size * self.vocab_size
        return embed_tflops, embed_proj_tflops

    def get_seq_tflops(self, seqlen, causal=False):
        scale = 0.5 if causal else 1
        config = self
        hidden_size = config.hidden_size
        num_heads = config.num_heads
        dim_head = config.dim_head
        embed_tflops, embed_proj_tflops = self.get_emb_tflops(seqlen)
        ffn_tflops = self.get_ffn_tflops(seqlen)
        attn_proj_tflops = 2 * seqlen * 3 * hidden_size * (dim_head * num_heads)
        attn_qk_tflops = 2 * seqlen * seqlen * dim_head * num_heads * scale
        attn_softmax_tflops = 3 * seqlen * seqlen * num_heads + 2 * seqlen * seqlen * num_heads * dim_head
        attn_softmax_tflops *= scale
        attn_o_proj_tflops = 2 * seqlen * hidden_size * (dim_head * num_heads)
        attn_linear = attn_proj_tflops + attn_o_proj_tflops
        attn_quadratic = attn_qk_tflops + attn_softmax_tflops
        total = (
            embed_tflops
            + config.num_layers * (attn_linear + ffn_tflops)
            + config.get_quadratic_layers() * attn_quadratic
            + config.deltanet_layers * config.get_deltanet_core_tflops(seqlen)
            + embed_proj_tflops
        )
        return total / 10 ** 12

    def get_prefix_tflops(self, seqlen, prefix):
        attn_quadratic = seqlen * prefix * (self.dim_head * 4 + 3) \
            * self.num_heads - seqlen ** 2 * (4 * self.dim_head + 3) \
            * self.num_heads / 2
        attn_linear = seqlen * 8 * self.hidden_size * self.num_heads * self.dim_head
        ffn_tflops = self.get_ffn_tflops(seqlen)
        embed_tflops,emb_proj_tflops = self.get_emb_tflops(seqlen)
        tf = (
            embed_tflops
            + self.num_layers * (attn_linear + ffn_tflops)
            + self.get_quadratic_layers() * attn_quadratic
            + self.deltanet_layers * self.get_deltanet_core_tflops(seqlen)
            + emb_proj_tflops
        )
        return tf / 10 ** 12

class sp_queue:
    def __init__(self, pipe_sp_splits=4, print=False, chunk=None, add_msg=""):
        # two stage queue
        # first stage use offset to track the current queue
        # second stage use idx to track the current item
        self.queues = [[]]
        self.p = print
        self.c = chunk
        self.info = add_msg
        self._offset = 0
        self._idx = 0
        self.count = 0
        self.pipe_sp_splits = pipe_sp_splits
        self.tail_obj = None

    def __len__(self):
        return self.count

    def print_log(self,msg):
        if torch.distributed.get_rank() == 3 and self.p:
            print(f"{self.info} chunk {self.c}: "+msg)

    def append(self, obj):
        self.print_log("append inp")
        self.tail_obj = obj
        self.queues[self._offset].append(obj)
        self._idx += 1
        if self._idx == self.pipe_sp_splits:
            self.print_log("full queue , create new one")
            self.queues.append([])
            self._idx = 0
            self._offset += 1
        self.count += 1
    
    def pop(self, idx=0):
        self.print_log(f"pop head inp of first queue")
        assert idx == 0, "only pop head item"
        self.count -= 1
        if len(self.queues[0]) == 1:
            if self._offset > 0:
                self._offset -= 1
                return self.queues.pop(0)[0]
            else:
                return self.queues[0].pop(-1)
        else:
            return self.queues[0].pop(-1)

    def __getitem__(self, idx):
        self.print_log(f"get tail inp ")
        assert idx == -1
        return self.tail_obj
        

partitions = None

def _parse_layer_selection(spec):
    layers = set()
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        if '-' in item:
            start_s, end_s = item.split('-', 1)
            layers.update(range(int(start_s), int(end_s) + 1))
        else:
            layers.add(int(item))
    return layers

def _count_hybrid_softmax_layers(args):
    if not getattr(args, 'use_deltanet', False):
        return args.num_layers

    layers = set()
    explicit_layers = getattr(args, 'deltanet_hybrid_attention_layers', '')
    if explicit_layers.strip():
        layers.update(_parse_layer_selection(explicit_layers))

    period = getattr(args, 'deltanet_hybrid_attention_period', 0)
    if period > 0:
        offset = getattr(args, 'deltanet_hybrid_attention_offset', 0)
        for layer in range(1, args.num_layers + 1):
            if (layer - offset) % period == 0:
                layers.add(layer)

    return len([layer for layer in layers if 1 <= layer <= args.num_layers])

def _split_alignment(args):
    align = 128
    if args.sequence_parallel:
        tp = args.tensor_model_parallel_size
        align = abs(align * tp) // math.gcd(align, tp)
    if args.seq_length % align != 0:
        # Keep legacy behavior for unusual sequence lengths that cannot be
        # partitioned into fully aligned chunks.
        return args.tensor_model_parallel_size if args.sequence_parallel else 1
    return align

def _add_deltanet_cost_config(config, args, softmax_layers):
    if not getattr(args, 'use_deltanet', False):
        return

    deltanet_layers = args.num_layers - softmax_layers
    config.update({
        "deltanet_layers": max(deltanet_layers, 0),
        "deltanet_head_dim": (
            args.deltanet_head_dim
            if getattr(args, 'deltanet_head_dim', None) is not None
            else args.hidden_size // args.num_attention_heads
        ),
        "deltanet_chunk_size": 64,
        "deltanet_conv_size": getattr(args, 'deltanet_conv_size', 4),
        "deltanet_use_beta": getattr(args, 'deltanet_use_beta', True),
        "deltanet_use_short_conv": getattr(args, 'deltanet_use_short_conv', True),
        "deltanet_use_output_gate": getattr(args, 'deltanet_use_output_gate', False),
    })

def get_tflops():
    args = get_args()
    config = {
        "num_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "ffn_size": args.ffn_hidden_size,
        "num_heads": args.num_attention_heads,
        "dim_head": args.hidden_size // args.num_attention_heads,
        "vocab_size": args.padded_vocab_size
    }
    config = SeqTFlops(**config)
    tflops = config.get_seq_tflops(args.seq_length, causal=True)
    return tflops

    args.total_tflops = sol.total_tflops 
def get_splits():
    global partitions
    args = get_args()
    if args.pipe_sp_strategy == "average":
        return [args.seq_length // args.pipe_sp_splits] * args.pipe_sp_splits
    if args.pipe_sp_splits == 1:
        return [args.seq_length]
    if partitions is None:
        assert args is not None
        seqlen = args.seq_length
        config = {
            "num_layers": args.num_layers,
            "hidden_size": args.hidden_size,
            "ffn_size": args.ffn_hidden_size,
            "num_heads": args.num_attention_heads,
            "dim_head": args.hidden_size // args.num_attention_heads,
            "vocab_size": args.padded_vocab_size
        }
        if args.pipe_sp_strategy == "hybrid_comp":
            softmax_layers = _count_hybrid_softmax_layers(args)
            config["softmax_layers"] = softmax_layers
            _add_deltanet_cost_config(config, args, softmax_layers)
        tflops_config = SeqTFlops(**config)
        sol = solver(seqlen, tflops_config)
        mod = _split_alignment(args)
        partitions = sol.solve_partition(args.pipe_sp_splits, mod)
        return partitions
    else:
        return partitions


class sp_shape_queue:
    def __init__(self, seqlen, bs, sz, backward=False):
        self.splits = get_splits() 
        self.idx = 0 if not backward else len(self.splits) - 1
        self.shape = [[[s, bs, sz]] for s in self.splits] 
        args = get_args()
        if args.sequence_parallel:
            self.shape = [[[s // args.tensor_model_parallel_size, bs, sz]] for s in self.splits]
        self.backward = backward
    def __iter__(self):
        iter = self.shape[self.idx].__iter__()
        if not self.backward:
            self.idx = (self.idx +1) % len(self.splits)
        else:
            self.idx = (self.idx -1) % len(self.splits)
        return iter
    def get(self):
        res = self.shape[self.idx][0]
        if not self.backward:
            self.idx = (self.idx +1) % len(self.splits)
        else:
            self.idx = (self.idx -1) % len(self.splits)
        return res
