# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real multi-node MoRI dispatch + AITER GEMM + MoRI combine regression test
for the buffer-trim fix in rocm_aiter_moe.py / prepare_finalize/mori.py.

Drives MoriPrepareAndFinalize.prepare()/.finalize() and AiterExperts.apply()
directly, bypassing FusedMoEFactory (see test_moe_layer_multinode.py).

Launch, single node (world_size == GPU_PER_NODE, no real multi-host needed):

    torchrun --nnodes 1 --nproc-per-node 8 \\
      -m pytest -v -s kernels/moe/test_rocm_mori_moe_ep.py

Launch, genuine multi-node (one torchrun per host, same rendezvous, one host
as rank 0):

    MASTER_ADDR=<node0-ip> MASTER_PORT=29525 \\
    torchrun --nnodes 4 --nproc-per-node 8 --node-rank <0..3> \\
      --rdzv_backend=c10d --rdzv_endpoint=<node0-ip>:29525 \\
      -m pytest -v -s kernels/moe/test_rocm_mori_moe_ep.py
"""

import importlib.util
import os

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm() or importlib.util.find_spec("mori") is None,
    reason="requires ROCm and mori",
)

GPU_PER_NODE = 8
HIDDEN = 4096
INTER = 1024
TOPK = 8
EXPERTS_PER_RANK = 8
MAX_TOK = 128
NUM_BIG_SMALL_ROUNDS = 20


def _make_topk_ids(num_tokens, num_experts, topk):
    logits = torch.randn(num_tokens, num_experts, device="cuda")
    _, ids = torch.topk(torch.softmax(logits, dim=-1), k=topk, dim=-1)
    return ids.to(torch.int32)


def test_mori_dispatch_aiter_combine_trim_regression():
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    if world_size == 0:
        pytest.skip(
            "must be launched under torchrun (WORLD_SIZE unset) -- see "
            "this file's module docstring"
        )
    if world_size < GPU_PER_NODE:
        pytest.skip(
            f"world_size={world_size} < {GPU_PER_NODE} -- need at least one "
            "full 8-GPU node"
        )

    import mori
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(local_rank)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    cpu_group = dist.new_group(list(range(world_size)), backend="gloo")
    torch._C._distributed_c10d._register_process_group("mori", cpu_group)
    mori.shmem.shmem_torch_process_group_init("mori")

    from vllm.config import VllmConfig
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FUSED_MOE_UNQUANTIZED_CONFIG,
    )
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
        AiterExperts,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize.mori import (
        MoriPrepareAndFinalize,
    )

    from .test_rocm_aiter_moe import _assert_aiter_supported, _shuffle_moe_weights
    from .utils import make_dummy_moe_config

    _assert_aiter_supported()

    num_experts = EXPERTS_PER_RANK * world_size
    # Single real node: split it into 2 logical MoRI "nodes" so InterNodeV1
    # traffic (and the bug it can trigger) still occurs; genuine multi-node
    # already has real nodes, so use the real per-node count.
    gpu_per_node = GPU_PER_NODE // 2 if world_size == GPU_PER_NODE else GPU_PER_NODE
    mori_config = mori.ops.EpDispatchCombineConfig(
        data_type=torch.bfloat16,
        rank=rank,
        world_size=world_size,
        hidden_dim=HIDDEN,
        scale_dim=0,
        scale_type_size=0,
        max_token_type_size=2,
        max_num_inp_token_per_rank=MAX_TOK,
        num_experts_per_rank=EXPERTS_PER_RANK,
        num_experts_per_token=TOPK,
        warp_num_per_block=16,
        block_num=32,
        rdma_block_num=16,
        kernel_type=mori.ops.EpDispatchCombineKernelType.InterNodeV1,
        gpu_per_node=gpu_per_node,
    )
    op = mori.ops.EpDispatchCombineOp(mori_config)
    pf = MoriPrepareAndFinalize(
        op, max_tokens_per_rank=MAX_TOK, num_dispatchers=world_size
    )

    torch.manual_seed(rank + 1)
    w1 = torch.randn(
        EXPERTS_PER_RANK, INTER * 2, HIDDEN, dtype=torch.bfloat16, device="cuda"
    ) / (HIDDEN**0.5)
    w2 = torch.randn(
        EXPERTS_PER_RANK, HIDDEN, INTER, dtype=torch.bfloat16, device="cuda"
    ) / (INTER**0.5)
    w1s, w2s = _shuffle_moe_weights(w1, w2)

    moe_config = make_dummy_moe_config(
        num_experts=num_experts,
        num_local_experts=EXPERTS_PER_RANK,
        experts_per_token=TOPK,
        hidden_dim=HIDDEN,
        intermediate_size=INTER,
        in_dtype=torch.bfloat16,
        max_num_tokens=MAX_TOK,
    )
    experts = AiterExperts(
        moe_config=moe_config, quant_config=FUSED_MOE_UNQUANTIZED_CONFIG
    )
    vllm_config = VllmConfig()

    def round_trip(n_real, marker):
        a1 = torch.zeros(MAX_TOK, HIDDEN, dtype=torch.bfloat16, device="cuda")
        ids = torch.zeros(MAX_TOK, TOPK, dtype=torch.int32, device="cuda")
        weights = torch.zeros(MAX_TOK, TOPK, dtype=torch.float32, device="cuda")
        if n_real > 0:
            a1[:n_real].fill_(marker)
            ids[:n_real] = _make_topk_ids(n_real, num_experts, TOPK)
            weights[:n_real] = 1.0 / TOPK

        (
            dispatch_a1,
            dispatch_scale,
            expert_tokens_meta,
            dispatch_ids,
            dispatch_weights,
        ) = pf.prepare(
            a1, weights, ids, num_experts, None, False, FUSED_MOE_UNQUANTIZED_CONFIG
        )

        m = dispatch_a1.shape[0]
        ws13, ws2, out_shape = experts.workspace_shapes(
            M=m,
            N=INTER,
            K=HIDDEN,
            topk=TOPK,
            global_num_experts=num_experts,
            local_num_experts=EXPERTS_PER_RANK,
            expert_tokens_meta=expert_tokens_meta,
            activation=MoEActivation.SILU,
        )
        workspace13 = torch.empty(ws13, dtype=torch.bfloat16, device="cuda")
        workspace2 = torch.empty(ws2, dtype=torch.bfloat16, device="cuda")
        aiter_out = torch.full(
            out_shape, float("nan"), dtype=torch.bfloat16, device="cuda"
        )

        with set_forward_context(None, vllm_config, num_tokens=m):
            experts.apply(
                output=aiter_out,
                hidden_states=dispatch_a1,
                w1=w1s,
                w2=w2s,
                topk_weights=dispatch_weights,
                topk_ids=dispatch_ids,
                activation=MoEActivation.SILU,
                global_num_experts=num_experts,
                expert_map=None,
                a1q_scale=dispatch_scale,
                a2_scale=None,
                workspace13=workspace13,
                workspace2=workspace2,
                expert_tokens_meta=expert_tokens_meta,
                apply_router_weight_on_input=False,
            )

        final_out = torch.zeros(MAX_TOK, HIDDEN, dtype=torch.bfloat16, device="cuda")
        pf.finalize(final_out, aiter_out, weights, ids, False, None)
        torch.accelerator.synchronize()
        dist.barrier()
        return final_out[:n_real].float() if n_real > 0 else final_out[:0].float()

    # warmup, then repeated (big, small) cycles reusing the same mori_op /
    # dispatch buffers -- a big round must precede a small round to trigger
    # the bug.
    round_trip(MAX_TOK, 5.0)
    for _ in range(NUM_BIG_SMALL_ROUNDS):
        round_trip(MAX_TOK, 99.0)
        small_out = round_trip(4, 1.0)
        assert torch.isfinite(small_out).all()
        assert small_out.abs().sum() > 0, (
            "MoRI dispatch/combine returned all-zero output for real "
            "tokens after buffer reuse -- the dispatch-output trim in "
            "MoriPrepareAndFinalize.prepare() may be missing"
        )

    dist.barrier()
    dist.destroy_process_group()


def _set_dp_token_counts(monkeypatch, mori_pf, counts, rank):
    """Make the forward context report `counts` tokens per DP rank (None: no
    DP metadata) and this process DP rank `rank`."""
    from types import SimpleNamespace

    dp_metadata = (
        None
        if counts is None
        else SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor(counts))
    )
    monkeypatch.setattr(mori_pf, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        mori_pf, "get_forward_context", lambda: SimpleNamespace(dp_metadata=dp_metadata)
    )
    monkeypatch.setattr(
        mori_pf, "get_dp_group", lambda: SimpleNamespace(rank_in_group=rank)
    )


@pytest.mark.parametrize(
    "counts, rank, input_num_tokens, expected",
    [
        ([3, 5, 0, 4], 1, 5, 12),  # the sum of the per-rank token counts
        ([3, 5, 0, 4], 1, 6, None),  # this rank dispatches rows it did not count
        ([3, 5], 1, 5, None),  # the counts do not cover the EP group
        (None, 0, 5, None),  # no DP metadata
    ],
)
def test_dp_rows_bound(monkeypatch, counts, rank, input_num_tokens, expected):
    from vllm.model_executor.layers.fused_moe.prepare_finalize import mori as mori_pf

    _set_dp_token_counts(monkeypatch, mori_pf, counts, rank)
    assert mori_pf._dp_rows_bound(4, input_num_tokens) == expected


def test_dp_rows_bound_without_forward_context(monkeypatch):
    from vllm.model_executor.layers.fused_moe.prepare_finalize import mori as mori_pf

    monkeypatch.setattr(mori_pf, "is_forward_context_available", lambda: False)
    assert mori_pf._dp_rows_bound(4, 5) is None


@pytest.mark.parametrize(
    "capturing, counts, recv, expected_rows",
    [
        # Captured: trimmed to the DP-count bound, 12 of 5 x 8 x 4 = 160 rows.
        (True, [3, 5, 0, 4], None, 12),
        # Captured without matching counts: tokens x topk x EP, no trim.
        (True, [3, 6, 0, 4], None, 160),
        # Eager: the exact received count, below the DP-count bound.
        (False, [3, 5, 0, 4], 7, 7),
    ],
)
def test_trim_dispatch_output_rows(monkeypatch, capturing, counts, recv, expected_rows):
    from vllm.model_executor.layers.fused_moe.prepare_finalize import mori as mori_pf

    _set_dp_token_counts(monkeypatch, mori_pf, counts, 1)
    monkeypatch.setattr(mori_pf, "_is_stream_capturing", lambda: capturing)
    rows = 5 * 8 * 4
    a1, scale, ids, weights = mori_pf._trim_dispatch_output(
        torch.zeros(rows, 16),
        torch.zeros(rows, 2),
        torch.zeros(rows, 8, dtype=torch.int32),
        torch.zeros(rows, 8),
        None if recv is None else torch.tensor([recv], dtype=torch.int32),
        ep_size=4,
        topk=8,
        input_num_tokens=5,
    )
    assert scale is not None
    assert [t.shape[0] for t in (a1, scale, ids, weights)] == [expected_rows] * 4
