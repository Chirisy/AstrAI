"""Backend selection and context-manager switching tests.

These tests do not require CUDA — they only check that the active
backend is correctly set and restored.
"""

import pytest

from astrai.extension import (
    ATTN_BACKEND,
    AttentionBackendFactory,
    CudaBackend,
    FlashInferBackend,
    attn_backend,
    get_backend,
)


def test_default_backend_resolves_to_available():
    """Default backend is the first available in cuda > flashinfer > flash > torch order."""
    from astrai.extension.attention_backend import (
        CudaBackend,
        FlashAttnBackend,
        FlashInferBackend,
        TorchNativeBackend,
    )

    backend = get_backend()
    assert isinstance(
        backend, (CudaBackend, FlashInferBackend, FlashAttnBackend, TorchNativeBackend)
    )


def test_attn_backend_context_with_enum():
    default = get_backend()
    with attn_backend(ATTN_BACKEND.CUDA):
        assert isinstance(get_backend(), CudaBackend)
    assert get_backend() is default


def test_attn_backend_context_with_registered_name():
    default = get_backend()
    with attn_backend("cuda"):
        assert isinstance(get_backend(), CudaBackend)
    assert get_backend() is default


def test_attn_backend_context_with_flashinfer_enum():
    default = get_backend()
    with attn_backend(ATTN_BACKEND.FLASHINFER):
        assert isinstance(get_backend(), FlashInferBackend)
    assert get_backend() is default


def test_attention_backend_factory_lists_builtin_backends():
    assert AttentionBackendFactory.list_registered() == [
        "cuda",
        "flash",
        "flashinfer",
        "torch_native",
    ]


def test_attn_backend_rejects_unknown_registered_name():
    with pytest.raises(ValueError, match="Unknown component: 'unknown'"):
        with attn_backend("unknown"):
            pass


def test_attn_backend_context_with_class():
    default = get_backend()
    with attn_backend(CudaBackend):
        assert isinstance(get_backend(), CudaBackend)
    assert get_backend() is default


def test_attn_backend_context_with_instance():
    custom = CudaBackend()
    default = get_backend()
    with attn_backend(custom):
        assert get_backend() is custom
    assert get_backend() is default


def test_cudabackend_is_context_manager():
    default = get_backend()
    with CudaBackend():
        assert isinstance(get_backend(), CudaBackend)
    assert get_backend() is default
