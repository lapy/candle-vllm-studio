# Candle-vLLM Studio UI Migration Guide

## Issue
The current Model Config UI contains many fields from llama.cpp that don't exist in candle-vllm.

## Candle-vLLM Actual CLI Parameters

Based on `backend/candle_manager.py` lines 246-298, candle-vllm accepts:

```bash
candle-vllm \
  --w /path/to/weights \      # weights_path (REQUIRED)
  --host 0.0.0.0 \             # host
  --p 41234 \                  # port (REQUIRED)
  --mem 4096 \                 # kvcache_mem_gpu (MB)
  --dtype f16 \                # Runtime dtype (f16, bf16, f32, or null for auto)
  --isq q6k \                  # In-situ quantization
  --max-gen-tokens 4096 \      # Maximum generated tokens
  --d 0                        # Device ID (GPU index)
```

## Fields to KEEP in UI

**Essential Tab:**
1. ✓ Candle Build (dropdown with showClear - override system default)
2. ✓ Weights Directory (readonly/disabled - managed by app)
3. ✓ Host (text input)
4. ✓ Port (number input, 1-65535)
5. ✓ KV Cache GPU Memory (number input, 0-64 GB)
6. ✓ Runtime DType (dropdown: Auto/FP16/BF16/FP32)
7. ✓ In-situ Quantization (ISQ dropdown)
8. ✓ Max Generated Tokens (number input)

**Advanced Tab (NEW):**
9. ✓ Device ID (number input, 0-15)
10. ✓ Timeout (number input, health check timeout)
11. ✓ Cargo Features (text input for comma-separated features)
12. ✓ Extra CLI Arguments (textarea for additional args)
13. ✓ Environment Variables (textarea for KEY=VALUE pairs)
14. ✓ Build Profile (dropdown: release/debug)

## Fields to REMOVE

All of these are llama.cpp-specific and NOT supported by candle-vllm:

### Memory & Context Tab (REMOVE ENTIRE TAB)
- ❌ GPU Layers (n_gpu_layers)
- ❌ Main GPU (main_gpu)
- ❌ Tensor Split (tensor_split)
- ❌ Context Size (ctx_size)
- ❌ Batch Size (batch_size)
- ❌ U-Batch Size (ubatch_size)
- ❌ No Memory Map (no_mmap)
- ❌ Mlock (mlock)
- ❌ Low VRAM (low_vram)
- ❌ CPU Threads (threads, threads_batch)

### Generation Tab (REMOVE ENTIRE TAB)
- ❌ Max Predict (n_predict)
- ❌ Temperature (temp/temperature)
- ❌ Top-K (top_k)
- ❌ Top-P (top_p)
- ❌ Repeat Penalty (repeat_penalty)
- ❌ Min-P (min_p)
- ❌ Typical-P (typical_p)
- ❌ TFS-Z (tfs_z)
- ❌ Presence Penalty (presence_penalty)
- ❌ Frequency Penalty (frequency_penalty)
- ❌ Mirostat Mode/Tau/Eta (mirostat_*)
- ❌ Seed (seed)
- ❌ Stop Words (stop)
- ❌ Grammar (grammar)
- ❌ JSON Schema (json_schema)
- ❌ Jinja Template (jinja)

**Reason:** Candle-vllm handles these via OpenAI-compatible API requests at inference time, NOT at server startup.

### Performance Tab (REMOVE ENTIRE TAB)
- ❌ Parallel (parallel)
- ❌ Flash Attention checkbox (flash_attn) - build-time feature, not runtime
- ❌ Continuous Batching (cont_batching)
- ❌ No KV Offload (no_kv_offload)
- ❌ Logits All (logits_all)
- ❌ Embedding Mode (embedding)
- ❌ K Cache Type (cache_type_k)
- ❌ V Cache Type (cache_type_v)
- ❌ MoE Expert Offloading (moe_offload_pattern, moe_offload_custom)

### GPU Tab (KEEP but simplify)
- Keep GPU detection/info display
- Remove configuration options that don't exist

### RoPE Tab (REMOVE ENTIRE TAB)
- ❌ RoPE Frequency Base (rope_freq_base)
- ❌ RoPE Frequency Scale (rope_freq_scale)
- ❌ YARN Extension Factor (yarn_ext_factor)
- ❌ YARN Attention Factor (yarn_attn_factor)
- ❌ RoPE Scaling Type (rope_scaling)

### Advanced/Custom Tab (SIMPLIFY)
- Keep: features, extra_args, env, build_profile
- Remove: custom YAML, custom args that reference unsupported params

## Implementation Plan

1. ✅ Make weights_path readonly/disabled
2. ✅ Add showClear to build selector with updated help text
3. ⚠️ Remove all unsupported tabs and fields
4. ⚠️ Create simplified "Advanced" tab with only supported fields
5. ⚠️ Update getDefaultConfig() to remove unsupported fields
6. ⚠️ Update smart-auto backend to only set supported fields
7. ⚠️ Update presets to only include supported fields
8. ⚠️ Add info message explaining that generation parameters are set via API
9. ⚠️ Add device_id field
10. ⚠️ Update validation logic to remove checks for unsupported fields

## User-Facing Message

Add a prominent info banner at the top of the config:

> **Note:** Candle-vllm is designed differently from llama.cpp. Generation parameters (temperature, top-k, sampling, etc.) are configured per-request via the OpenAI-compatible API, not at server startup. This UI configures only the runtime server settings.

## Backend Updates Needed

1. Remove unsupported fields from smart-auto calculations
2. Remove unsupported fields from architecture presets
3. Update model config storage to ignore legacy fields
4. Add device_id to runtime config

## Migration Strategy

Since this is a major breaking change:
1. Keep existing configs in database but ignore unsupported fields
2. Show migration notice for models with old configs
3. Provide "Clean Config" button to remove unsupported fields

