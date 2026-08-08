"""Attention backend abstraction with context-manager switching.

The backend encapsulates KV cache I/O and attention computation. The
attention module (GQA/MLA) keeps projections, rotary, QK-norm, gating,
and output projection; the backend handles everything from "write K/V
to cache" through "SDPA output".

Usage — mirroring ``torch.nn.attention.sdpa_kernel``:

    from astrai.extension import attn_backend, ATTN_BACKEND

    with attn_backend(ATTN_BACKEND.TORCH_NATIVE):
        engine.generate("hello")

    # or with an instance:
    with attn_backend(TorchNativeBackend()):
        ...

    # or the shorthand (instance is itself a context manager):
    with TorchNativeBackend():
        ...

Thread-safe via ``contextvars`` — each scheduler thread gets its own
active backend. ``get_backend()`` returns the active one, falling back
to a process-wide default (cuda > flashinfer > flash > torch, overridable via
``ASTR_BACKEND``).

Layout convention: all q/k/v are ``[batch, seq_len, n_heads, head_dim]``
(blhd). The backend returns ``[batch, seq_len, n_heads * head_dim]``.
"""

import contextvars
import enum
import functools
import importlib
import os
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from astrai.extension.attention_ops import (
    attn_paged_decode,
    attn_paged_prefill,
)
from astrai.extension.loader import is_available
from astrai.factory import BaseFactory

if TYPE_CHECKING:
    from astrai.inference.core.cache import KVCache

_current_backend: contextvars.ContextVar["AttentionBackend"] = contextvars.ContextVar(
    "attn_backend"
)


@functools.lru_cache(maxsize=1)
def flash_attn_available() -> bool:
    if not torch.cuda.is_available():
        return False
    fa = _get_flash_attn()
    if fa is None:
        return False

    try:
        major = int(fa.__version__.split(".")[0])
        cc = torch.cuda.get_device_capability()
        cc_num = cc[0] * 10 + cc[1]
    except Exception:
        major, cc_num = 0, 0
    if (major >= 3 and cc_num < 90) or (major < 3 and 0 < cc_num < 70):
        return False

    try:
        if not hasattr(fa, "flash_attn_func"):
            return False
        x = torch.zeros(1, 1, 1, 64, device="cuda", dtype=torch.bfloat16)
        out = fa.flash_attn_func(x, x, x, causal=True)
        return bool(torch.isfinite(out).all().item())
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def _get_flash_attn():
    try:
        return importlib.import_module("flash_attn")
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def flashinfer_available() -> bool:
    if not torch.cuda.is_available():
        return False
    fi = _get_flashinfer()
    if fi is None:
        return False
    return all(
        hasattr(fi, name)
        for name in (
            "BatchDecodeWithPagedKVCacheWrapper",
            "BatchPrefillWithPagedKVCacheWrapper",
        )
    )


@functools.lru_cache(maxsize=1)
def _get_flashinfer():
    try:
        return importlib.import_module("flashinfer")
    except Exception:
        return None


class ATTN_BACKEND(enum.Enum):
    """Backend selector enum, mirroring ``torch.nn.attention.SDPBackend``."""

    TORCH_NATIVE = "torch_native"
    CUDA = "cuda"
    FLASHINFER = "flashinfer"
    FLASH = "flash"


_default_backend: Optional["AttentionBackend"] = None
_default_backend_lock = threading.Lock()


def _priority_backends() -> list["AttentionBackend"]:
    """Available backends in priority order: cuda -> flashinfer -> flash -> torch."""
    backends: list[AttentionBackend] = []
    if is_available("attn_paged_decode") and is_available("attn_paged_prefill"):
        backends.append(CudaBackend())
    if flashinfer_available():
        backends.append(FlashInferBackend())
    if flash_attn_available():
        backends.append(FlashAttnBackend())
    backends.append(TorchNativeBackend())
    return backends


def _backend_supports(
    backend: "AttentionBackend",
    q: Tensor,
    k: Tensor,
    kv_cache: Optional["KVCache"],
    attn_mask: Optional[Tensor],
    is_causal: bool,
) -> bool:
    """Whether ``backend`` can run this attention call.

    The CUDA kernels are bf16-only, support head_dim in 32/64/128/256, and
    need a KV cache (decode/prefill); everything else falls back to torch.
    """
    if isinstance(backend, CudaBackend):
        return (
            kv_cache is not None
            and q.dtype == torch.bfloat16
            and q.size(-1) in (32, 64, 128, 256)
        )
    if isinstance(backend, FlashInferBackend):
        if not flashinfer_available() or kv_cache is None:
            return False
        if q.dtype not in (torch.float16, torch.bfloat16):
            return False
        if q.size(-1) != k.size(-1) or q.size(2) % k.size(2) != 0:
            return False
        if kv_cache.kv_indptr is None:
            return False
        if q.size(1) > 1:
            if kv_cache.qo_indptr is None:
                return False
            if attn_mask is not None and attn_mask.dim() not in (2, 3, 4):
                return False
        return True
    if isinstance(backend, FlashAttnBackend):
        if not flash_attn_available():
            return False
        if q.dtype not in (torch.float16, torch.bfloat16):
            return False
        if q.size(1) == 1 and kv_cache is not None:
            return True
        return attn_mask is None
    return True


def _resolve_default_backend() -> "AttentionBackend":
    """Pick the highest-priority available backend (cuda -> flashinfer -> flash -> torch).

    Set ``ASTR_BACKEND`` to override: ``ASTR_BACKEND=cuda``, ``torch_native``,
    ``flashinfer``, or ``flash``.  The value is the registered name (same as the
    ``ATTN_BACKEND`` enum value).

    Resolved lazily on first ``get_backend()`` and cached.  Per-call
    capability fallback happens in ``attention()``, so the default is
    safe for training and fp32 models.
    """
    forced = os.environ.get("ASTR_BACKEND", "").strip().lower()
    if forced:
        try:
            return AttentionBackendFactory.create(forced)
        except (ValueError, RuntimeError):
            pass
    return _priority_backends()[0]


def get_backend() -> "AttentionBackend":
    """Return the active backend for the current thread/context.

    Falls back to the highest-priority available backend (cuda -> flashinfer ->
    flash -> torch_native) when no backend has been activated via ``with``.  Set
    ``ASTR_BACKEND`` to override the default.
    """
    try:
        return _current_backend.get()
    except LookupError:
        global _default_backend
        if _default_backend is None:
            with _default_backend_lock:
                if _default_backend is None:
                    _default_backend = _resolve_default_backend()
        return _default_backend


@contextmanager
def attn_backend(backend: Union[str, ATTN_BACKEND, "AttentionBackend", type]):
    """Context manager to select an attention backend.

    Mirrors ``torch.nn.attention.sdpa_kernel``. Accepts an
    registered name, ``ATTN_BACKEND`` enum value, backend class, or instance.

    Examples::

        with attn_backend(ATTN_BACKEND.TORCH_NATIVE):
            ...
        with attn_backend(TorchNativeBackend):
            ...
        with attn_backend(TorchNativeBackend()):
            ...
    """
    if isinstance(backend, ATTN_BACKEND):
        instance = AttentionBackendFactory.create(backend.value)
    elif isinstance(backend, str):
        instance = AttentionBackendFactory.create(backend)
    elif isinstance(backend, type) and issubclass(backend, AttentionBackend):
        instance = backend()
    elif isinstance(backend, AttentionBackend):
        instance = backend
    else:
        raise TypeError(
            f"expected a registered name, ATTN_BACKEND, AttentionBackend type, "
            f"or instance, "
            f"got {type(backend).__name__}"
        )
    token = _current_backend.set(instance)
    try:
        yield instance
    finally:
        _current_backend.reset(token)


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Expand KV heads to match Q heads for GQA."""
    bs, slen, n_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_heads, n_rep, head_dim)
        .reshape(bs, slen, n_heads * n_rep, head_dim)
    )


def _write_and_gather_kv(
    kv_cache: "KVCache",
    k: Tensor,
    v: Tensor,
    layer_id: int,
    q: Tensor,
    attn_mask: Optional[Tensor],
) -> tuple[Tensor, Tensor]:
    kv_cache.k_buffer[layer_id, kv_cache.out_cache_loc] = k
    kv_cache.v_buffer[layer_id, kv_cache.out_cache_loc] = v
    max_len = kv_cache.max_len
    indices = kv_cache.req_to_token[kv_cache.req_pool_indices, :max_len]
    if q.size(1) == 1 and attn_mask is not None and attn_mask.dim() == 4:
        pos_mask = attn_mask[:, 0, 0]
    else:
        pos_mask = (
            torch.arange(max_len, device=q.device)[None, :] < kv_cache.seq_lens[:, None]
        )
    indices = torch.where(pos_mask, indices, torch.zeros_like(indices))
    return kv_cache.k_buffer[layer_id, indices], kv_cache.v_buffer[layer_id, indices]


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kv_cache: Optional["KVCache"] = None,
    layer_id: int = 0,
    attn_mask: Optional[Tensor] = None,
    is_causal: bool = False,
) -> Tensor:
    """Functional attention entry point — mirrors ``F.scaled_dot_product_attention``.

    Delegates to the active backend (set via ``with attn_backend(...)``).
    Handles KV cache I/O, GQA head expansion, and causal masking so the
    caller only needs to provide projected q/k/v.

    Args:
        q: [batch, q_len, n_heads, head_dim] (blhd)
        k: [batch, q_len, n_kv_heads, head_dim] (blhd)
        v: [batch, q_len, n_kv_heads, head_dim] (blhd)
        kv_cache: cache dataclass, or None for training (no cache).
        layer_id: transformer layer index for buffer access.
        attn_mask: pre-built attention mask (SDPA-compatible).
        is_causal: whether to apply causal masking.

    Returns:
        [batch, q_len, n_heads * head_dim]
    """
    backend = get_backend()
    if not _backend_supports(backend, q, k, kv_cache, attn_mask, is_causal):
        try:
            explicit = _current_backend.get()
        except LookupError:
            explicit = None
        if explicit is not None:
            raise RuntimeError(
                f"Explicitly-set backend {type(backend).__name__} cannot "
                f"handle this attention call (shape={q.shape}, "
                f"dtype={q.dtype}, kv_cache={'none' if kv_cache is None else 'present'}, "
                f"attn_mask={'none' if attn_mask is None else 'present'}). "
                f"Remove the attn_backend() context or switch to a compatible backend."
            )
        for candidate in _priority_backends():
            if isinstance(candidate, type(backend)):
                continue
            if _backend_supports(candidate, q, k, kv_cache, attn_mask, is_causal):
                backend = candidate
                break
    return backend.forward(q, k, v, kv_cache, layer_id, attn_mask, is_causal)


class AttentionBackend(ABC):
    """Abstract base for attention computation strategies.

    Subclasses implement ``fwd_decode`` (q_len == 1, with cache) and
    ``fwd_prefill`` (q_len > 1, with or without cache). The public
    ``forward`` method dispatches based on q_len.

    Three equivalent ways to activate a backend::

        with attn_backend(ATTN_BACKEND.TORCH_NATIVE):  # enum
            ...
        with attn_backend(TorchNativeBackend):          # class
            ...
        with TorchNativeBackend():                       # instance
            ...
    """

    def __enter__(self) -> "AttentionBackend":
        self._token = _current_backend.set(self)
        return self

    def __exit__(self, *exc) -> None:
        _current_backend.reset(self._token)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: Optional["KVCache"],
        layer_id: int,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        """Dispatch to decode or extend based on q_len.

        Args:
            q: [batch, q_len, n_heads, head_dim]
            k: [batch, q_len, n_kv_heads, head_dim]
            v: [batch, q_len, n_kv_heads, head_dim]
            kv_cache: cache dataclass, or None for training (no cache).
            layer_id: transformer layer index for buffer access.
            attn_mask: pre-built attention mask compatible with SDPA.
            is_causal: whether to apply causal masking.

        Returns:
            [batch, q_len, n_heads * head_dim]
        """
        if kv_cache is not None and q.size(1) == 1:
            return self.fwd_decode(q, k, v, kv_cache, layer_id, attn_mask, is_causal)
        return self.fwd_prefill(q, k, v, kv_cache, layer_id, attn_mask, is_causal)

    @abstractmethod
    def fwd_decode(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: Optional["KVCache"],
        layer_id: int,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        """Single-token decode with KV cache."""

    @abstractmethod
    def fwd_prefill(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: Optional["KVCache"],
        layer_id: int,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        """Multi-token prefill or training forward."""

    @staticmethod
    def supports_graph() -> bool:
        """Return True if this backend supports CUDA-graph capture.

        Override in subclasses that can run under ``torch.cuda.graph``.

        Called on the *active* backend instance (or its class) — a cheap
        boolean check with no side-effects.
        """
        return False


class AttentionBackendFactory(BaseFactory[AttentionBackend]):
    """Factory for registered attention backends."""


@AttentionBackendFactory.register(ATTN_BACKEND.TORCH_NATIVE.value)
class TorchNativeBackend(AttentionBackend):
    """Reference backend using torch SDPA with indirect KV cache indexing.

    Writes new K/V into the cache buffers, gathers the full sequence K/V
    via ``req_to_token`` indirect indexing, then calls
    ``F.scaled_dot_product_attention``.

    For training (``kv_cache is None``), skips cache I/O entirely and
    runs SDPA directly on the projected q/k/v.
    """

    @staticmethod
    def supports(**kwargs) -> bool:
        return True

    def fwd_decode(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: Optional["KVCache"],
        layer_id: int,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        return self._forward(q, k, v, kv_cache, layer_id, attn_mask, is_causal)

    def fwd_prefill(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: Optional["KVCache"],
        layer_id: int,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        return self._forward(q, k, v, kv_cache, layer_id, attn_mask, is_causal)

    def _forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: Optional["KVCache"],
        layer_id: int,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        if kv_cache is not None:
            k, v = _write_and_gather_kv(kv_cache, k, v, layer_id, q, attn_mask)

        n_rep = q.size(2) // k.size(2)
        if n_rep > 1:
            k = repeat_kv(k, n_rep)
            v = repeat_kv(v, n_rep)

        out = F.scaled_dot_product_attention(
            q.permute(0, 2, 1, 3),
            k.permute(0, 2, 1, 3),
            v.permute(0, 2, 1, 3),
            attn_mask,
            is_causal=is_causal,
        )
        out = out.permute(0, 2, 1, 3).contiguous().flatten(2)
        return out


@AttentionBackendFactory.register(ATTN_BACKEND.CUDA.value)
class CudaBackend(AttentionBackend):
    """CUDA kernel backend with direct KV cache access.

    Decode path: writes K/V to the flat pool, then calls
    ``attn_paged_decode`` with req_to_token + kv_indptr.

    Prefill path: writes K/V to the flat pool, then calls
    ``attn_paged_prefill`` with ragged-batch support via qo_indptr +
    kv_indptr.

    ``kv_cache is None`` (training) raises — the per-call fallback to
    torch SDPA for training / fp32 / unsupported head_dim happens in the
    ``attention()`` entry point.

    Raises ``RuntimeError`` if the required kernel is not available.
    """

    @staticmethod
    def supports(**kwargs) -> bool:
        head_dim = kwargs.get("head_dim", -1)
        return (
            torch.cuda.is_available()
            and head_dim in (32, 64, 128, 256)
            and is_available("attn_paged_decode")
            and is_available("attn_paged_prefill")
        )

    @staticmethod
    def supports_graph() -> bool:
        return True

    def fwd_decode(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: Optional["KVCache"],
        layer_id: int,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        if kv_cache is None:
            raise RuntimeError("CudaBackend does not support training (kv_cache=None)")

        loc = kv_cache.out_cache_loc[:, 0]
        kv_cache.k_buffer[layer_id].index_copy_(0, loc, k[:, 0])
        kv_cache.v_buffer[layer_id].index_copy_(0, loc, v[:, 0])

        q_3d = q.squeeze(1)

        kv_indptr = kv_cache.kv_indptr

        out = attn_paged_decode(
            q_3d,
            kv_cache.k_buffer[layer_id],
            kv_cache.v_buffer[layer_id],
            kv_cache.req_to_token,
            kv_cache.req_pool_indices,
            kv_indptr,
            kv_cache.max_len,
            is_causal=True,
            o_part_buf=kv_cache.decode_o_part,
            ml_part_buf=kv_cache.decode_ml_part,
            out_buf=kv_cache.decode_out,
        )
        return out.unsqueeze(1).flatten(2)

    def fwd_prefill(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: Optional["KVCache"],
        layer_id: int,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        if kv_cache is None:
            raise RuntimeError("CudaBackend does not support training (kv_cache=None)")

        loc = kv_cache.out_cache_loc.reshape(-1)
        kv_cache.k_buffer[layer_id].index_copy_(
            0, loc, k.reshape(-1, k.size(2), k.size(3))
        )
        kv_cache.v_buffer[layer_id].index_copy_(
            0, loc, v.reshape(-1, v.size(2), v.size(3))
        )

        b = q.size(0)
        q_len = q.size(1)

        kv_indptr = kv_cache.kv_indptr
        qo_indptr = kv_cache.qo_indptr

        q_flat = q.reshape(b * q_len, q.size(2), q.size(3))

        out = attn_paged_prefill(
            q_flat,
            kv_cache.k_buffer[layer_id],
            kv_cache.v_buffer[layer_id],
            kv_cache.req_to_token,
            kv_cache.req_pool_indices,
            kv_indptr,
            qo_indptr,
            attn_mask,
            q_len,
            is_causal=is_causal,
        )
        return out.reshape(b, q_len, q.size(2), q.size(3)).flatten(2)


@AttentionBackendFactory.register(ATTN_BACKEND.FLASHINFER.value)
class FlashInferBackend(AttentionBackend):
    """FlashInfer backend using paged KV wrappers.

    The project KV pool is token-addressed, so it is exposed to FlashInfer
    as a paged cache with ``page_size=1``. New K/V values are written to the
    existing flat cache buffers, and FlashInfer receives the per-request
    indptr plus flattened physical token indices from ``req_to_token``.
    """

    _PAGE_SIZE = 1

    def __init__(self, workspace_size: int = 128 * 1024 * 1024):
        self.workspace_size = workspace_size
        self.float_workspace_buffer: Optional[Tensor] = None
        self.prefill_wrapper: Any = None
        self.decode_wrapper: Any = None
        self._device: Optional[torch.device] = None
        self._use_tensor_cores: Optional[bool] = None
        self._ones_cpu = torch.empty(0, dtype=torch.int32)
        self._last_decode_cache: Optional["KVCache"] = None
        self._last_decode_signature: Optional[tuple] = None
        self._last_decode_tensors: tuple[Tensor, ...] = ()
        self._last_prefill_cache: Optional["KVCache"] = None
        self._last_prefill_signature: Optional[tuple] = None
        self._last_prefill_tensors: tuple[Tensor, ...] = ()

    @staticmethod
    def supports(**kwargs) -> bool:
        head_dim = kwargs.get("head_dim")
        if head_dim is not None and head_dim <= 0:
            return False
        return flashinfer_available()

    def fwd_decode(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: Optional["KVCache"],
        layer_id: int,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        if kv_cache is None:
            raise RuntimeError(
                "FlashInferBackend does not support training (kv_cache=None)"
            )
        self._ensure_initialized(q.device, self._should_use_tensor_cores(q, k))
        self._write_decode_kv(kv_cache, k, v, layer_id)
        self._plan_decode(q, k, kv_cache)

        assert self.decode_wrapper is not None
        out = self._run_wrapper(
            self.decode_wrapper,
            q.squeeze(1).contiguous(),
            self._paged_kv_cache(kv_cache, layer_id),
        )
        out = self._unwrap_flashinfer_output(out)
        return out.unsqueeze(1).contiguous().flatten(2)

    def fwd_prefill(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: Optional["KVCache"],
        layer_id: int,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        if kv_cache is None:
            raise RuntimeError(
                "FlashInferBackend does not support training (kv_cache=None)"
            )
        if kv_cache.qo_indptr is None:
            raise RuntimeError("FlashInferBackend requires qo_indptr for prefill")

        self._ensure_initialized(q.device, self._should_use_tensor_cores(q, k))
        self._write_prefill_kv(kv_cache, k, v, layer_id)
        q_lens = self._indptr_lens(kv_cache.qo_indptr)
        self._plan_prefill(q, k, kv_cache, attn_mask, is_causal, q_lens)

        assert self.prefill_wrapper is not None
        q_flat = self._flatten_prefill_q(q, q_lens)
        out = self._run_wrapper(
            self.prefill_wrapper,
            q_flat.contiguous(),
            self._paged_kv_cache(kv_cache, layer_id),
        )
        out = self._unwrap_flashinfer_output(out)
        out = self._restore_prefill_output(out, q, q_lens)
        return out.contiguous().flatten(2)

    def _ensure_initialized(self, device: torch.device, use_tensor_cores: bool) -> None:
        device = torch.device(device)
        if self.float_workspace_buffer is not None:
            if self._device != device:
                raise RuntimeError(
                    "FlashInferBackend cannot move devices after initialization"
                )
            return

        fi = _get_flashinfer()
        if fi is None:
            raise RuntimeError(
                "FlashInferBackend requires the optional flashinfer-python package. "
                "Install it with `pip install flashinfer-python`."
            )

        self._device = device
        self._use_tensor_cores = use_tensor_cores
        self.float_workspace_buffer = torch.zeros(
            self.workspace_size, dtype=torch.uint8, device=device
        )
        self.prefill_wrapper = fi.BatchPrefillWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            backend="fa2",
        )
        self.decode_wrapper = fi.BatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            use_tensor_cores=use_tensor_cores,
            kv_layout="NHD",
            backend="fa2",
        )

        int_workspace = getattr(self.prefill_wrapper, "_int_workspace_buffer", None)
        if int_workspace is not None and hasattr(
            self.decode_wrapper, "_int_workspace_buffer"
        ):
            self.decode_wrapper._int_workspace_buffer = int_workspace

    @staticmethod
    def _should_use_tensor_cores(q: Tensor, k: Tensor) -> bool:
        return q.size(2) // k.size(2) >= 4

    @staticmethod
    def _as_int32(tensor: Tensor) -> Tensor:
        if tensor.dtype == torch.int32:
            return tensor
        return tensor.to(torch.int32)

    def _get_ones_cpu(self, batch: int) -> Tensor:
        if batch <= self._ones_cpu.numel():
            return self._ones_cpu[:batch]
        size = 1 << (batch - 1).bit_length()
        try:
            self._ones_cpu = torch.ones(size, dtype=torch.int32, pin_memory=True)
        except RuntimeError:
            self._ones_cpu = torch.ones(size, dtype=torch.int32)
        return self._ones_cpu[:batch]

    @staticmethod
    def _unwrap_flashinfer_output(out):
        if isinstance(out, tuple):
            return out[0]
        return out

    @staticmethod
    def _run_wrapper(wrapper, q: Tensor, paged_kv_cache: tuple[Tensor, Tensor]):
        try:
            return wrapper.run(q=q, paged_kv_cache=paged_kv_cache, return_lse=False)
        except TypeError as exc:
            if "return_lse" not in str(exc):
                raise
            return wrapper.run(q=q, paged_kv_cache=paged_kv_cache)

    @staticmethod
    def _write_decode_kv(
        kv_cache: "KVCache", k: Tensor, v: Tensor, layer_id: int
    ) -> None:
        loc = kv_cache.out_cache_loc[:, 0]
        kv_cache.k_buffer[layer_id].index_copy_(0, loc, k[:, 0])
        kv_cache.v_buffer[layer_id].index_copy_(0, loc, v[:, 0])

    @staticmethod
    def _write_prefill_kv(
        kv_cache: "KVCache", k: Tensor, v: Tensor, layer_id: int
    ) -> None:
        loc = kv_cache.out_cache_loc.reshape(-1)
        kv_cache.k_buffer[layer_id].index_copy_(
            0, loc, k.reshape(-1, k.size(2), k.size(3))
        )
        kv_cache.v_buffer[layer_id].index_copy_(
            0, loc, v.reshape(-1, v.size(2), v.size(3))
        )

    @staticmethod
    def _paged_kv_cache(kv_cache: "KVCache", layer_id: int) -> tuple[Tensor, Tensor]:
        n_kv = kv_cache.k_buffer.size(2)
        head_dim = kv_cache.k_buffer.size(3)
        return (
            kv_cache.k_buffer[layer_id].view(-1, 1, n_kv, head_dim),
            kv_cache.v_buffer[layer_id].view(-1, 1, n_kv, head_dim),
        )

    def _paged_indices(self, kv_cache: "KVCache") -> Tensor:
        rows = kv_cache.req_to_token[kv_cache.req_pool_indices, : kv_cache.max_len]
        mask = (
            torch.arange(kv_cache.max_len, device=rows.device)[None, :]
            < kv_cache.seq_lens[:, None]
        )
        return rows[mask].to(torch.int32)

    @staticmethod
    def _indptr_lens(indptr: Tensor) -> list[int]:
        values = indptr.detach().to("cpu").tolist()
        return [int(values[i + 1] - values[i]) for i in range(len(values) - 1)]

    def _flatten_custom_mask(
        self, attn_mask: Tensor, q: Tensor, kv_cache: "KVCache", q_lens: list[int]
    ) -> Tensor:
        mask = attn_mask
        if mask.device != q.device:
            mask = mask.to(q.device)
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)
        if mask.dim() == 4:
            if mask.size(1) != 1:
                raise ValueError(
                    "FlashInferBackend expects attention mask head dimension to be 1"
                )
            mask = mask[:, 0]
        elif mask.dim() == 2:
            mask = mask[:, None, :].expand(-1, q.size(1), -1)
        elif mask.dim() != 3:
            raise ValueError(
                f"unsupported attention mask shape for FlashInfer: {tuple(mask.shape)}"
            )

        kv_lens = self._indptr_lens(kv_cache.kv_indptr)
        max_q = max(q_lens, default=0)
        max_kv = max(kv_lens, default=0)
        if mask.size(0) < len(q_lens) or mask.size(1) < max_q or mask.size(2) < max_kv:
            raise ValueError(
                "attention mask is smaller than FlashInfer q/kv lengths: "
                f"mask={tuple(mask.shape)}, q_lens={q_lens}, kv_lens={kv_lens}"
            )
        pieces = [
            mask[i, : q_lens[i], : kv_lens[i]].reshape(-1) for i in range(len(q_lens))
        ]
        if not pieces:
            return mask.new_empty((0,), dtype=torch.bool)
        return torch.cat(pieces).contiguous()

    @staticmethod
    def _flatten_prefill_q(q: Tensor, q_lens: list[int]) -> Tensor:
        if all(length == q.size(1) for length in q_lens):
            return q.reshape(q.size(0) * q.size(1), q.size(2), q.size(3))
        return torch.cat([q[i, : q_lens[i]] for i in range(len(q_lens))], dim=0)

    @staticmethod
    def _restore_prefill_output(out: Tensor, q: Tensor, q_lens: list[int]) -> Tensor:
        if all(length == q.size(1) for length in q_lens):
            return out.reshape(q.size(0), q.size(1), q.size(2), q.size(3))
        restored = q.new_zeros((q.size(0), q.size(1), q.size(2), q.size(3)))
        offset = 0
        for i, length in enumerate(q_lens):
            restored[i, :length] = out[offset : offset + length]
            offset += length
        return restored

    def _plan_decode(self, q: Tensor, k: Tensor, kv_cache: "KVCache") -> None:
        assert kv_cache.kv_indptr is not None
        batch = q.size(0)
        signature = (
            q.size(2),
            k.size(2),
            q.size(3),
            q.dtype,
            kv_cache.k_buffer.dtype,
            kv_cache.max_len,
            kv_cache.req_pool_indices.data_ptr(),
            kv_cache.seq_lens.data_ptr(),
            kv_cache.kv_indptr.data_ptr(),
        )
        if (
            self._last_decode_cache is kv_cache
            and self._last_decode_signature == signature
        ):
            return

        indptr = self._as_int32(kv_cache.kv_indptr)
        indices = self._paged_indices(kv_cache)
        last_page_len = self._get_ones_cpu(batch)
        seq_lens = self._as_int32(kv_cache.seq_lens)

        assert self.decode_wrapper is not None
        self.decode_wrapper.plan(
            indptr=indptr,
            indices=indices,
            last_page_len=last_page_len,
            num_qo_heads=q.size(2),
            num_kv_heads=k.size(2),
            head_dim=q.size(3),
            page_size=self._PAGE_SIZE,
            pos_encoding_mode="NONE",
            seq_lens=seq_lens,
            data_type=kv_cache.k_buffer.dtype,
            q_data_type=q.dtype,
            kv_data_type=kv_cache.k_buffer.dtype,
            non_blocking=True,
        )
        self._last_decode_cache = kv_cache
        self._last_decode_signature = signature
        self._last_decode_tensors = (indptr, indices, last_page_len, seq_lens)

    def _plan_prefill(
        self,
        q: Tensor,
        k: Tensor,
        kv_cache: "KVCache",
        attn_mask: Optional[Tensor],
        is_causal: bool,
        q_lens: list[int],
    ) -> None:
        assert kv_cache.kv_indptr is not None and kv_cache.qo_indptr is not None
        mask_signature = (
            (attn_mask.data_ptr(), tuple(attn_mask.shape), attn_mask.dtype)
            if attn_mask is not None
            else None
        )
        signature = (
            q.size(2),
            k.size(2),
            q.size(3),
            q.dtype,
            kv_cache.k_buffer.dtype,
            kv_cache.max_len,
            bool(is_causal and attn_mask is None),
            kv_cache.req_pool_indices.data_ptr(),
            kv_cache.seq_lens.data_ptr(),
            kv_cache.kv_indptr.data_ptr(),
            kv_cache.qo_indptr.data_ptr(),
            mask_signature,
        )
        if (
            self._last_prefill_cache is kv_cache
            and self._last_prefill_signature == signature
        ):
            return

        custom_mask = (
            self._flatten_custom_mask(attn_mask, q, kv_cache, q_lens)
            if attn_mask is not None
            else None
        )

        qo_indptr = self._as_int32(kv_cache.qo_indptr)
        kv_indptr = self._as_int32(kv_cache.kv_indptr)
        indices = self._paged_indices(kv_cache)
        last_page_len = self._get_ones_cpu(q.size(0))
        seq_lens = self._as_int32(kv_cache.seq_lens)

        assert self.prefill_wrapper is not None
        self.prefill_wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=indices,
            paged_kv_last_page_len=last_page_len,
            num_qo_heads=q.size(2),
            num_kv_heads=k.size(2),
            head_dim_qk=q.size(3),
            page_size=self._PAGE_SIZE,
            custom_mask=custom_mask,
            causal=bool(is_causal and custom_mask is None),
            pos_encoding_mode="NONE",
            seq_lens=seq_lens,
            q_data_type=q.dtype,
            kv_data_type=kv_cache.k_buffer.dtype,
            non_blocking=True,
        )
        self._last_prefill_cache = kv_cache
        self._last_prefill_signature = signature
        tensors = [qo_indptr, kv_indptr, indices, last_page_len, seq_lens]
        if custom_mask is not None:
            tensors.append(custom_mask)
        self._last_prefill_tensors = tuple(tensors)


@AttentionBackendFactory.register(ATTN_BACKEND.FLASH.value)
class FlashAttnBackend(AttentionBackend):
    """FlashAttention backend via the optional ``flash-attn`` package.

    Decode (q_len=1, contiguous cache): uses ``flash_attn_with_kvcache``,
    which reads K/V directly from the flat pool via cache_batch_idx +
    cache_seqlens — no materialized KV gather.

    Prefill / non-contiguous decode: falls back to KV gather +
    ``flash_attn_func``.
    """

    @staticmethod
    def supports(**kwargs) -> bool:
        return flash_attn_available()

    def fwd_decode(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: Optional["KVCache"],
        layer_id: int,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        return self._forward(q, k, v, kv_cache, layer_id, attn_mask, is_causal)

    def fwd_prefill(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: Optional["KVCache"],
        layer_id: int,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        return self._forward(q, k, v, kv_cache, layer_id, attn_mask, is_causal)

    def _forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: Optional["KVCache"],
        layer_id: int,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        if kv_cache is not None:
            if q.size(1) == 1 and kv_cache.k_buffer.size(
                1
            ) == kv_cache.req_to_token.size(0) * kv_cache.req_to_token.size(1):
                return self._decode_with_kvcache(q, k, v, kv_cache, layer_id)
            k, v = _write_and_gather_kv(kv_cache, k, v, layer_id, q, attn_mask)

        n_rep = q.size(2) // k.size(2)
        if n_rep > 1:
            k = repeat_kv(k, n_rep)
            v = repeat_kv(v, n_rep)

        if attn_mask is not None and not is_causal:
            raise ValueError(
                "FlashAttnBackend does not support a custom attention mask; "
                "use a causal mask or select TorchNativeBackend."
            )
        fa = _get_flash_attn()
        if fa is None:
            raise RuntimeError(
                "FlashAttnBackend requires the optional 'flash-attn' package. "
                "Install with `pip install flash-attn`."
            )
        out = fa.flash_attn_func(
            q.contiguous(), k.contiguous(), v.contiguous(), causal=is_causal
        )
        return out.contiguous().flatten(2)

    def _decode_with_kvcache(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: "KVCache",
        layer_id: int,
    ) -> Tensor:
        max_batch = kv_cache.req_to_token.size(0)
        max_seq = kv_cache.req_to_token.size(1)
        n_kv = k.size(2)

        k_cache = kv_cache.k_buffer[layer_id].view(max_batch, max_seq, n_kv, k.size(3))
        v_cache = kv_cache.v_buffer[layer_id].view(max_batch, max_seq, n_kv, v.size(3))

        fa = _get_flash_attn()
        out = fa.flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            k=k,
            v=v,
            cache_seqlens=(kv_cache.seq_lens - 1).to(torch.int32),
            cache_batch_idx=kv_cache.req_pool_indices.to(torch.int32),
            causal=True,
        )
        return out.flatten(2)
