import pytest
from typing import Dict

from backend.smart_auto import SmartAutoConfig


class DummyModel:
    def __init__(self, name: str, file_size: int, quantization: str | None = None, file_path: str = ""):
        self.name = name
        self.file_size = file_size
        self.quantization = quantization
        self.file_path = file_path


def _gpu_info(total_gb: float, free_gb: float, compute_capability: str = "8.9"):
    return {
        "vendor": "nvidia",
        "cpu_only_mode": False,
        "gpus": [
            {
                "index": 0,
                "name": "Test GPU",
                "compute_capability": compute_capability,
                "memory": {
                    "total": int(total_gb * (1024 ** 3)),
                    "free": int(free_gb * (1024 ** 3)),
                    "used": int((total_gb - free_gb) * (1024 ** 3)),
                },
            }
        ],
    }


def _nvlink_gpu_info(total_gb: float = 80, free_gb: float = 72) -> Dict:
    total_bytes = int(total_gb * (1024 ** 3))
    free_bytes = int(free_gb * (1024 ** 3))
    gpu_template = lambda index: {
        "index": index,
        "name": f"GPU{index}",
        "compute_capability": "9.0",
        "memory": {
            "total": total_bytes,
            "free": free_bytes,
            "used": total_bytes - free_bytes,
        },
        "topology": {
            "numa_node": 0,
            "cpu_affinity": [index],
            "nvlink_links": [
                {
                    "peer": 1 - index,
                    "link_id": 0,
                    "version": 3,
                    "speed_gbps": 50.0,
                    "p2p_supported": True,
                }
            ],
        },
        "pcie": {"max_generation": 4, "max_link_width": 16},
    }

    return {
        "vendor": "nvidia",
        "cpu_only_mode": False,
        "gpus": [gpu_template(0), gpu_template(1)],
        "topology": {
            "ancestor_matrix": {"0": {"0": "self", "1": "internal"}, "1": {"0": "internal", "1": "self"}},
            "nvlink_clusters": [[0, 1]],
            "isolated_gpus": [],
            "mixed_topology": False,
        },
    }


@pytest.mark.asyncio
async def test_generate_config_gpu_defaults():
    model = DummyModel("example-chat", file_size=7 * 1024 ** 3, quantization="q4k")
    gpu_info = _gpu_info(total_gb=80, free_gb=72)

    config = await SmartAutoConfig().generate_config(
        model=model,
        gpu_info=gpu_info,
        usage_mode="multi_user",
        speed_quality=45,
        use_case="chat",
    )

    assert config.get("dtype") in (None, "f16", "bf16", "f32")
    assert config["kvcache_mem_gpu"] > 0
    assert config["max_gen_tokens"] >= 512
    assert config["usage_mode"] == "multi_user"
    assert config["max_num_seqs"] is None or config["max_num_seqs"] >= 1
    assert config["prefill_chunk_size"] % 1024 == 0
    assert config["health_endpoints"] == ["/v1/models"]
    assert config["health_interval"] == 30
    assert config["model_name"] == model.name
    assert "topology_plan" in config
    assert config["device_ids"] == [0]


@pytest.mark.asyncio
async def test_generate_config_cpu_only_allocates_cpu_cache():
    model = DummyModel("example-cpu", file_size=4 * 1024 ** 3)
    gpu_info = {
        "vendor": None,
        "cpu_only_mode": True,
        "gpus": [],
    }

    config = await SmartAutoConfig().generate_config(
        model=model,
        gpu_info=gpu_info,
        usage_mode="single_user",
        speed_quality=70,
    )

    assert config["cpu"] is True
    assert config["kvcache_mem_gpu"] == 0
    assert config["kvcache_mem_cpu"] is not None
    assert config["kvcache_mem_cpu"] > 0
    assert config["max_num_seqs"] is None
    assert config["device_ids"] is None


@pytest.mark.asyncio
async def test_generate_config_respects_overrides():
    model = DummyModel("example-override", file_size=6 * 1024 ** 3)
    gpu_info = _gpu_info(total_gb=48, free_gb=40, compute_capability="9.0")

    config = await SmartAutoConfig().generate_config(
        model=model,
        gpu_info=gpu_info,
        target_concurrency=20,
        max_tokens_hint=1024,
        speed_quality=30,
    )

    assert config["max_num_seqs"] == 16  # clamped at backend limit
    assert config["max_gen_tokens"] == 1024
    assert config["kvcache_mem_gpu"] > 0
    assert config["topology_plan"]["global"]["recommended_max_num_seqs"] >= 1


@pytest.mark.asyncio
async def test_generate_config_nvlink_restriction():
    model = DummyModel("nvlink-model", file_size=12 * 1024 ** 3)
    gpu_info = _nvlink_gpu_info()

    config = await SmartAutoConfig().generate_config(
        model=model,
        gpu_info=gpu_info,
        restrict_to_nvlink=True,
        speed_quality=40,
    )

    plan = config["topology_plan"]
    assert plan["global"]["restrict_to_nvlink"] is True
    assert plan["groups"], "Expected at least one group in topology plan"
    assert len(plan["groups"]) == 1
    nvlink_group = plan["groups"][0]
    assert nvlink_group["tp_degree"] == 2
    assert nvlink_group["link_type"] == "nvlink"
    device_indices = {device["index"] for device in nvlink_group["devices"]}
    assert device_indices == {0, 1}
    assert config["device_ids"] == [0, 1]
    assert plan["global"].get("selected_devices") == [0, 1]


@pytest.mark.asyncio
async def test_gguf_allows_tensor_parallel():
    model = DummyModel(
        "gguf-model",
        file_size=6 * 1024 ** 3,
        quantization="Q4_K_M",
        file_path="Qwen3-4B-Q4_K_M.gguf",
    )
    gpu_info = _nvlink_gpu_info()

    config = await SmartAutoConfig().generate_config(
        model=model,
        gpu_info=gpu_info,
        speed_quality=45,
        use_case="chat",
    )

    plan = config["topology_plan"]
    assert plan["groups"], "Expected topology plan groups for gguf model"
    assert plan["global"]["tp_strategy"] == "tensor_parallel"
    assert config["device_ids"] == [0, 1]

