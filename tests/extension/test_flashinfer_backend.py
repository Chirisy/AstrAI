import torch

from astrai.extension import FlashInferBackend
from astrai.inference.core.cache import KVCache


class _FakeWrapper:
    def __init__(self):
        self.plan_calls = []
        self.run_calls = []

    def plan(self, **kwargs):
        self.plan_calls.append(kwargs)

    def run(self, *, q, paged_kv_cache, return_lse=False):
        self.run_calls.append((q, paged_kv_cache, return_lse))
        return q


def _backend() -> tuple[FlashInferBackend, _FakeWrapper, _FakeWrapper]:
    backend = FlashInferBackend(workspace_size=1)
    decode = _FakeWrapper()
    prefill = _FakeWrapper()
    backend.float_workspace_buffer = torch.empty(1)
    backend._device = torch.device("cpu")
    backend.decode_wrapper = decode
    backend.prefill_wrapper = prefill
    return backend, decode, prefill


def _kv_cache(
    *,
    seq_lens: list[int],
    out_cache_loc: torch.Tensor,
    kv_indptr: list[int],
    qo_indptr: list[int] | None = None,
) -> KVCache:
    return KVCache(
        k_buffer=torch.zeros(1, 12, 2, 8),
        v_buffer=torch.zeros(1, 12, 2, 8),
        req_to_token=torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.long),
        req_pool_indices=torch.tensor([0, 1], dtype=torch.long),
        seq_lens=torch.tensor(seq_lens, dtype=torch.long),
        out_cache_loc=out_cache_loc,
        max_len=max(seq_lens),
        kv_indptr=torch.tensor(kv_indptr, dtype=torch.int32),
        qo_indptr=(
            torch.tensor(qo_indptr, dtype=torch.int32)
            if qo_indptr is not None
            else None
        ),
    )


def test_flashinfer_decode_writes_cache_and_reuses_plan():
    backend, decode, _ = _backend()
    cache = _kv_cache(
        seq_lens=[3, 2],
        out_cache_loc=torch.tensor([[2], [5]], dtype=torch.long),
        kv_indptr=[0, 3, 5],
    )
    q = torch.randn(2, 1, 4, 8)
    k = torch.randn(2, 1, 2, 8)
    v = torch.randn(2, 1, 2, 8)

    out = backend.fwd_decode(q, k, v, cache, layer_id=0)
    backend.fwd_decode(q, k, v, cache, layer_id=0)

    assert out.shape == (2, 1, 32)
    torch.testing.assert_close(cache.k_buffer[0, [2, 5]], k[:, 0])
    torch.testing.assert_close(cache.v_buffer[0, [2, 5]], v[:, 0])
    assert len(decode.plan_calls) == 1
    assert decode.plan_calls[0]["indices"].tolist() == [0, 1, 2, 4, 5]
    assert decode.plan_calls[0]["last_page_len"].tolist() == [1, 1]
    assert decode.plan_calls[0]["page_size"] == 1


def test_flashinfer_prefill_flattens_custom_mask():
    backend, _, prefill = _backend()
    cache = _kv_cache(
        seq_lens=[3, 2],
        out_cache_loc=torch.tensor([[0, 1, 2], [4, 5, 6]], dtype=torch.long),
        kv_indptr=[0, 3, 5],
        qo_indptr=[0, 3, 6],
    )
    q = torch.randn(2, 3, 4, 8)
    k = torch.randn(2, 3, 2, 8)
    v = torch.randn(2, 3, 2, 8)
    mask = torch.tensor(
        [
            [
                [
                    [True, False, False],
                    [True, True, False],
                    [True, True, True],
                ]
            ],
            [
                [
                    [True, False, False],
                    [True, True, False],
                    [True, True, False],
                ]
            ],
        ]
    )

    out = backend.fwd_prefill(q, k, v, cache, layer_id=0, attn_mask=mask)

    plan = prefill.plan_calls[0]
    expected_mask = torch.cat(
        [mask[0, 0, :, :3].reshape(-1), mask[1, 0, :, :2].reshape(-1)]
    )
    assert out.shape == (2, 3, 32)
    assert plan["paged_kv_indices"].tolist() == [0, 1, 2, 4, 5]
    assert torch.equal(plan["custom_mask"], expected_mask)
    assert plan["causal"] is False
    assert plan["page_size"] == 1
