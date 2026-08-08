"""CUDA attention kernel wrappers with torch fallback.

Public API:
    - ``attn_decode`` — single-query decode attention
    - ``attn_prefill`` — multi-query prefill attention
    - ``attn_paged_decode`` — paged decode attention (direct page-table access)
    - ``AttentionBackend`` — ABC for attention computation strategies
    - ``TorchNativeBackend`` — default SDPA backend with KV cache I/O
    - ``CudaBackend`` — CUDA kernel backend with paged decode + prefill

Layout convention: all q/k/v are ``[batch, seq_len, n_heads, head_dim]``
(blhd). Scale is always ``1/sqrt(head_dim)``.

Each wrapper calls its compiled CUDA kernel directly. Fallback to torch
SDPA is handled by the attention backend, not the wrapper functions.
"""

from astrai.extension.attention_backend import (
    ATTN_BACKEND,
    AttentionBackend,
    AttentionBackendFactory,
    CudaBackend,
    FlashAttnBackend,
    FlashInferBackend,
    TorchNativeBackend,
    attention,
    attn_backend,
    get_backend,
)
from astrai.extension.attention_ops import (
    TensorLayout,
    attn_decode,
    attn_paged_decode,
    attn_prefill,
)
from astrai.extension.loader import KERNEL_NAMES, is_available
from astrai.extension.rotary_backend import apply_rotary_emb

__all__ = [
    "ATTN_BACKEND",
    "KERNEL_NAMES",
    "AttentionBackend",
    "AttentionBackendFactory",
    "CudaBackend",
    "FlashAttnBackend",
    "FlashInferBackend",
    "TensorLayout",
    "TorchNativeBackend",
    "apply_rotary_emb",
    "attention",
    "attn_backend",
    "attn_decode",
    "attn_paged_decode",
    "attn_prefill",
    "get_backend",
    "is_available",
]
