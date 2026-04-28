import torch

from tensorrt_llm._torch.modules.fused_moe.fused_moe_b12x_cpp import (
    B12xCppMoEWrapper,
    convert_sf_to_mma_layout,
)


def test_b12x_cpp_wrapper_converts_mma_scales_and_calls_op(monkeypatch):
    calls = {}

    def fake_b12x_moe_cuda(
        x,
        w1_weight,
        w1_sf_storage,
        w1_alpha,
        fc2_input_scale,
        w2_weight,
        w2_sf_storage,
        w2_alpha,
        topk_ids,
        topk_weights,
        output,
        num_experts,
        top_k,
        num_local_experts,
        activation,
    ):
        calls.update(
            w1_sf_storage=w1_sf_storage,
            w2_sf_storage=w2_sf_storage,
            output=output,
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=num_local_experts,
            activation=activation,
        )
        output.zero_()
        return output

    monkeypatch.setattr(
        torch.ops.trtllm,
        "b12x_moe_cuda",
        fake_b12x_moe_cuda,
        raising=False,
    )

    num_experts = 2
    hidden_size = 64
    intermediate_size = 128
    top_k = 2
    x = torch.randn(3, hidden_size, dtype=torch.bfloat16)
    topk_ids = torch.zeros(3, top_k, dtype=torch.int32)
    topk_weights = torch.ones(3, top_k, dtype=torch.float32)
    w1_weight = torch.zeros(num_experts, intermediate_size, hidden_size // 2, dtype=torch.uint8)
    w2_weight = torch.zeros(num_experts, hidden_size, intermediate_size // 2, dtype=torch.uint8)
    w1_sf = torch.arange(num_experts * 128 * 4, dtype=torch.uint8).reshape(num_experts * 128, 4)
    w2_sf = torch.arange(num_experts * 128 * 8, dtype=torch.uint8).reshape(num_experts * 128, 8)
    w1_sf_mma = convert_sf_to_mma_layout(
        w1_sf, m=intermediate_size, k=hidden_size, num_groups=num_experts
    )
    w2_sf_mma = convert_sf_to_mma_layout(
        w2_sf, m=hidden_size, k=intermediate_size, num_groups=num_experts
    )

    wrapper = B12xCppMoEWrapper(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation="relu2",
    )
    out = wrapper.run(
        x=x,
        w1_weight=w1_weight,
        w1_weight_sf=w1_sf_mma,
        w1_alpha=torch.ones(num_experts, dtype=torch.float32),
        w2_weight=w2_weight,
        w2_weight_sf=w2_sf_mma,
        w2_alpha=torch.ones(num_experts, dtype=torch.float32),
        fc2_input_scale=torch.ones(1, dtype=torch.float32),
        token_selected_experts=topk_ids,
        token_final_scales=topk_weights,
    )

    assert out.shape == (3, hidden_size)
    assert out.dtype == torch.bfloat16
    assert calls["output"] is out
    assert calls["w1_sf_storage"].shape == (num_experts * 128, 4)
    assert calls["w2_sf_storage"].shape == (num_experts * 128, 8)
    assert torch.equal(calls["w1_sf_storage"], w1_sf)
    assert torch.equal(calls["w2_sf_storage"], w2_sf)
    assert calls["num_experts"] == num_experts
    assert calls["top_k"] == top_k
    assert calls["num_local_experts"] == num_experts
    assert calls["activation"] == 1
