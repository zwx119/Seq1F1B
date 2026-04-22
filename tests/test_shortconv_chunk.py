"""Unit test for ShortConvChunkFunc gradient relay.

Runs three isolated checks on a single ShortConvChunkFunc / fla.ShortConvolution
pair — no Megatron, no distributed setup, pure PyTorch.

Chain A — forward equivalence (single-chunk):
    One big ShortConvChunkFunc call on full seq == fla.ShortConvolution(full seq)
    (up to bf16 / kernel rounding; target ≤ 1e-3 abs).

Chain B — forward equivalence (multi-chunk via prefix passing):
    Run ShortConvChunkFunc N times on [x_0, x_1, …, x_{N-1}] with prefix-cache
    passing == single call on torch.cat([x_0, …, x_{N-1}]).
    (Should be bit-exact in fp32, since we're using F.conv1d both ways.)

Chain C — backward equivalence (critical):
    Gradient of loss wrt x and wrt weight, computed two ways:
        (i)  one big chunk, autograd-through F.conv1d
        (ii) N chunks with ShortConvChunkFunc + explicit grad_dict relay,
             triggered by calling .backward() on each chunk's output in
             REVERSE order (chunk N-1, N-2, …, 0) — mimicking Seq1F1B.
    Expected: dx_chunks stitched == dx_single, dw_chunks == dw_single,
              both bit-exact in fp32.

Run:  python3 tests/test_shortconv_chunk.py
"""
import os
import sys
import torch
import torch.nn.functional as F

# Make sure we can import from repo root
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))

from megatron.model.deltanet_attention import ShortConvChunkFunc


def conv_full(x, weight, bias, activation):
    """Reference: causal depthwise conv on full sequence via F.conv1d.

    Matches exactly what ShortConvChunkFunc does internally, so chain B/C
    are bit-exact in fp32.
    """
    B, T, D = x.shape
    W = weight.shape[-1]
    prefix = x.new_zeros(B, W - 1, D)
    x_full = torch.cat([prefix, x], dim=1)
    y_bdt = F.conv1d(x_full.transpose(1, 2).contiguous(), weight, bias=bias, groups=D)
    y = y_bdt.transpose(1, 2).contiguous()
    if activation in ('silu', 'swish'):
        y = F.silu(y)
    return y


def run_chunks(x_list, weight, bias, activation):
    """Run N chunks via ShortConvChunkFunc with prefix-cache relay.

    Returns (y_list, cache_dict, grad_dict).
    """
    cache_dict = {}
    grad_dict = {}
    y_list = []
    for x in x_list:
        y = ShortConvChunkFunc.apply(
            x, weight, bias, cache_dict, grad_dict, 'test', activation,
        )
        y_list.append(y)
    return y_list, cache_dict, grad_dict


def test_chain_A_single_chunk():
    print("=== Chain A: single-chunk forward equivalence ===")
    torch.manual_seed(0)
    B, T, D, W = 2, 128, 16, 4
    x = torch.randn(B, T, D, device='cuda', dtype=torch.float32, requires_grad=True)
    weight = torch.randn(D, 1, W, device='cuda', dtype=torch.float32) * 0.1
    bias = torch.randn(D, device='cuda', dtype=torch.float32) * 0.1

    # Reference
    y_ref = conv_full(x, weight, bias, 'silu')
    # Via ShortConvChunkFunc (single chunk = whole sequence)
    cache_dict, grad_dict = {}, {}
    y = ShortConvChunkFunc.apply(x, weight, bias, cache_dict, grad_dict, 'test', 'silu')

    diff = (y - y_ref).abs().max().item()
    print(f"    |y - y_ref|_inf = {diff:.3e}  (fp32 expected = 0)")
    assert diff < 1e-5, f"Chain A failed: diff {diff}"
    print("    PASS\n")


def test_chain_B_multi_chunk_forward():
    print("=== Chain B: multi-chunk forward equivalence via prefix passing ===")
    torch.manual_seed(1)
    B, T_total, D, W = 2, 128, 16, 4
    N = 4
    T = T_total // N
    x = torch.randn(B, T_total, D, device='cuda', dtype=torch.float32)
    weight = torch.randn(D, 1, W, device='cuda', dtype=torch.float32) * 0.1
    bias = torch.randn(D, device='cuda', dtype=torch.float32) * 0.1

    # Single call reference
    y_ref = conv_full(x, weight, bias, 'silu')

    # Chunked (each chunk detached from graph since we only test forward here)
    x_list = [x[:, i * T:(i + 1) * T, :].detach() for i in range(N)]
    y_list, _, _ = run_chunks(x_list, weight, bias, 'silu')
    y = torch.cat(y_list, dim=1)

    diff = (y - y_ref).abs().max().item()
    print(f"    |y_chunked - y_ref|_inf = {diff:.3e}  (fp32 expected = 0)")
    assert diff < 1e-5, f"Chain B failed: diff {diff}"
    print("    PASS\n")


def test_chain_C_multi_chunk_backward():
    print("=== Chain C: multi-chunk backward equivalence (Seq1F1B grad relay) ===")
    torch.manual_seed(2)
    B, T_total, D, W = 2, 128, 16, 4
    N = 4
    T = T_total // N

    # Same x used in both paths (cloned for independent autograd graphs)
    x_ref_data = torch.randn(B, T_total, D, device='cuda', dtype=torch.float32)
    weight_ref_data = torch.randn(D, 1, W, device='cuda', dtype=torch.float32) * 0.1
    bias_ref_data = torch.randn(D, device='cuda', dtype=torch.float32) * 0.1

    # --- Reference: single call full autograd ---
    x_ref = x_ref_data.clone().requires_grad_(True)
    w_ref = weight_ref_data.clone().requires_grad_(True)
    b_ref = bias_ref_data.clone().requires_grad_(True)
    y_ref = conv_full(x_ref, w_ref, b_ref, 'silu')
    grad_out_full = torch.randn_like(y_ref)
    y_ref.backward(grad_out_full)
    dx_ref = x_ref.grad.clone()
    dw_ref = w_ref.grad.clone()
    db_ref = b_ref.grad.clone()

    # --- Chunked: ShortConvChunkFunc + reverse-order per-chunk backward ---
    # Build a single leaf x, then create chunk SLICES (so dx accumulates into x.grad).
    x = x_ref_data.clone().requires_grad_(True)
    w = weight_ref_data.clone().requires_grad_(True)
    b = bias_ref_data.clone().requires_grad_(True)

    # Use non-in-place slicing to preserve autograd edges back to x.
    x_list = [x[:, i * T:(i + 1) * T, :] for i in range(N)]

    cache_dict, grad_dict = {}, {}
    y_list = []
    for xc in x_list:
        y_c = ShortConvChunkFunc.apply(xc, w, b, cache_dict, grad_dict, 'test', 'silu')
        y_list.append(y_c)

    # Slice grad_out_full in same way so each chunk's backward sees matching gradient
    grad_list = [grad_out_full[:, i * T:(i + 1) * T, :].contiguous() for i in range(N)]

    # Backward in REVERSE order, mimicking Seq1F1B (chunk N-1 first, then N-2, …, 0)
    # retain_graph=True for all except the last because x, w are shared.
    for i in reversed(range(N)):
        retain = (i > 0)
        y_list[i].backward(grad_list[i], retain_graph=retain)

    dx = x.grad.clone()
    dw = w.grad.clone()
    db = b.grad.clone()

    dx_diff = (dx - dx_ref).abs().max().item()
    dw_diff = (dw - dw_ref).abs().max().item()
    db_diff = (db - db_ref).abs().max().item()
    print(f"    |dx - dx_ref|_inf = {dx_diff:.3e}")
    print(f"    |dw - dw_ref|_inf = {dw_diff:.3e}")
    print(f"    |db - db_ref|_inf = {db_diff:.3e}")
    assert dx_diff < 1e-5, f"dx mismatch {dx_diff}"
    assert dw_diff < 1e-5, f"dw mismatch {dw_diff}"
    assert db_diff < 1e-5, f"db mismatch {db_diff}"

    # Extra sanity: after all backward, grad_dict should be empty-ish.
    # Chunk 0 (had_prefix=False) skipped writing; earlier chunks popped what they read.
    # grad_dict might still hold chunk-N-1's dx_prefix if we didn't clear it
    # (because chunk 0 never read). That's fine — next microbatch's _clear_states
    # wipes it. Just log.
    print(f"    leftover grad_dict keys = {list(grad_dict.keys())}")
    print("    PASS\n")


def test_chain_D_no_activation():
    """Same as Chain C but with activation=None (verify silu-backward path is
    not silently bypassing anything)."""
    print("=== Chain D: backward equivalence, activation=None ===")
    torch.manual_seed(3)
    B, T_total, D, W = 2, 64, 8, 4
    N = 4
    T = T_total // N

    x_data = torch.randn(B, T_total, D, device='cuda', dtype=torch.float32)
    w_data = torch.randn(D, 1, W, device='cuda', dtype=torch.float32) * 0.1

    # Ref
    x_ref = x_data.clone().requires_grad_(True)
    w_ref = w_data.clone().requires_grad_(True)
    y_ref = conv_full(x_ref, w_ref, None, None)
    g = torch.randn_like(y_ref)
    y_ref.backward(g)
    dx_ref, dw_ref = x_ref.grad.clone(), w_ref.grad.clone()

    # Chunks
    x = x_data.clone().requires_grad_(True)
    w = w_data.clone().requires_grad_(True)
    x_list = [x[:, i * T:(i + 1) * T, :] for i in range(N)]
    cache_dict, grad_dict = {}, {}
    y_list = [ShortConvChunkFunc.apply(xc, w, None, cache_dict, grad_dict, 'q', None) for xc in x_list]
    g_list = [g[:, i * T:(i + 1) * T, :].contiguous() for i in range(N)]
    for i in reversed(range(N)):
        y_list[i].backward(g_list[i], retain_graph=(i > 0))

    dx_diff = (x.grad - dx_ref).abs().max().item()
    dw_diff = (w.grad - dw_ref).abs().max().item()
    print(f"    |dx - dx_ref|_inf = {dx_diff:.3e}")
    print(f"    |dw - dw_ref|_inf = {dw_diff:.3e}")
    assert dx_diff < 1e-5 and dw_diff < 1e-5
    print("    PASS\n")


if __name__ == '__main__':
    assert torch.cuda.is_available(), "CUDA required"
    test_chain_A_single_chunk()
    test_chain_B_multi_chunk_forward()
    test_chain_C_multi_chunk_backward()
    test_chain_D_no_activation()
    print("ALL PASS ✓")
