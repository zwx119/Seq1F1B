# Copyright (c) 2024, Seq1F1B-DeltaNet Contributors.

"""
DeltaNet Transformer Layer for Megatron Seq1F1B.

This module provides DeltaNetTransformerLayer, which is a drop-in replacement
for ParallelTransformerLayer. It replaces the softmax self-attention with
DeltaNet linear attention while keeping everything else the same:
  - Pre-LayerNorm architecture (input_layernorm + post_attention_layernorm)
  - ParallelMLP (SwiGLU or GELU)
  - Residual connections
  - Bias-dropout-add fusion
  - All Megatron pipeline/tensor parallelism infrastructure

The layer takes [s, b, h] input and returns [s, b, h] output, fully
compatible with the existing ParallelTransformer container.
"""

from contextlib import nullcontext
from typing import Optional

import torch

from megatron import get_args, core
from megatron.core import mpu, tensor_parallel
from megatron.model.module import MegatronModule
from megatron.model import LayerNorm
from megatron.model.enums import AttnMaskType, LayerType, AttnType
from megatron.model.transformer import (
    ParallelMLP,
    SwitchMLP,
    bias_dropout_add_fused_train,
    bias_dropout_add_fused_inference,
    get_bias_dropout_add,
    DropPath,
)
from megatron.model.deltanet_attention import DeltaNetAttention


class DeltaNetTransformerLayer(MegatronModule):
    """A single transformer layer using DeltaNet linear attention.

    This is structurally identical to ParallelTransformerLayer, but replaces
    ParallelAttention (softmax) with DeltaNetAttention (linear recurrent).

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.

    Key differences from ParallelTransformerLayer:
      - self.self_attention is DeltaNetAttention instead of ParallelAttention
      - No cross-attention support (DeltaNet is decoder-only for now)
      - No retro support
      - micro_sp_idx is passed through for Seq1F1B state management
    """

    def __init__(self, config, layer_number,
                 layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 drop_path_rate=0.):
        args = get_args()
        super(DeltaNetTransformerLayer, self).__init__()

        self.layer_number = layer_number
        self.layer_type = layer_type

        self.apply_residual_connection_post_layernorm = \
            config.apply_residual_connection_post_layernorm
        self.bf16 = config.bf16
        self.fp32_residual_connection = config.fp32_residual_connection

        # LayerNorm on the input data
        self.input_layernorm = LayerNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            no_persist_layer_norm=args.no_persist_layer_norm,
            sequence_parallel=config.sequence_parallel,
            apply_layernorm_1p=args.apply_layernorm_1p)

        # DeltaNet self-attention (replaces ParallelAttention)
        self.self_attention = DeltaNetAttention(config, layer_number)

        self.hidden_dropout = config.hidden_dropout
        self.bias_dropout_fusion = config.bias_dropout_fusion
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else None

        # LayerNorm on the attention output
        self.post_attention_layernorm = LayerNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            no_persist_layer_norm=not config.persist_layer_norm,
            sequence_parallel=config.sequence_parallel,
            apply_layernorm_1p=args.apply_layernorm_1p)

        # MLP
        if args.num_experts is not None:
            self.mlp = SwitchMLP(config)
        else:
            self.mlp = ParallelMLP(config)

        # Bias+dropout+add fusion handler
        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        self.bias_dropout_add_exec_handler = \
            nullcontext if use_nvfuser else torch.enable_grad

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                retriever_input=None,
                retriever_output=None,
                retriever_attn_mask=None,
                inference_params=None,
                rotary_pos_emb=None,
                micro_sp_idx=None):
        """
        Forward pass. Interface is identical to ParallelTransformerLayer.forward()
        so it's a drop-in replacement.

        Args:
            hidden_states: [s, b, h]
            attention_mask: unused by DeltaNet (kept for interface compatibility)
            micro_sp_idx: Seq1F1B split index (0, 1, ..., pipe_sp_splits-1)
            Others: kept for interface compatibility, mostly unused
        Returns:
            output: [s, b, h]
        """
        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        # DeltaNet self attention
        attention_output, attention_bias = self.self_attention(
            layernorm_output,
            attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            micro_sp_idx=micro_sp_idx)

        # Residual connection
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        if self.drop_path is None:
            if self.bias_dropout_fusion:
                if self.training:
                    bias_dropout_add_func = bias_dropout_add_fused_train
                else:
                    bias_dropout_add_func = bias_dropout_add_fused_inference
            else:
                bias_dropout_add_func = get_bias_dropout_add(self.training)

            if attention_bias is not None:
                attention_bias = attention_bias.expand_as(residual)
            with self.bias_dropout_add_exec_handler():
                layernorm_input = bias_dropout_add_func(
                    attention_output,
                    attention_bias,
                    residual,
                    self.hidden_dropout)
        else:
            out = torch.nn.functional.dropout(
                attention_output + (attention_bias if attention_bias is not None else 0),
                p=self.hidden_dropout,
                training=self.training)
            layernorm_input = residual + self.drop_path(out)

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP
        mlp_output, mlp_bias = self.mlp(layernorm_output)

        # Second residual connection
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        if self.drop_path is None:
            if mlp_bias is not None:
                mlp_bias = mlp_bias.expand_as(residual)
            with self.bias_dropout_add_exec_handler():
                output = bias_dropout_add_func(
                    mlp_output,
                    mlp_bias,
                    residual,
                    self.hidden_dropout)

            output = core.utils.make_viewless_tensor(
                inp=output,
                requires_grad=output.requires_grad,
                keep_graph=True)
        else:
            if mlp_bias is not None:
                mlp_output = mlp_output + mlp_bias
            out = torch.nn.functional.dropout(
                mlp_output,
                p=self.hidden_dropout,
                training=self.training)
            output = residual + self.drop_path(out)

        return output
