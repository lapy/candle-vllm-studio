# Candle vLLM Studio

An opinionated fork of the original llama-cpp-studio that replaces the llama.cpp backend with [candle-vllm](https://github.com/EricLBuehler/candle-vllm). The goal is to provide a local model manager with a familiar web UI while leaning on candle's OpenAI-compatible server.

## Features

- Hugging Face search & download with progress streaming
- Smart Auto configuration across runtime & sampling (speed/quality slider, use-case profiles, concurrency hints)
- Multi-GPU topology planning with NVLink detection, per-device budgeting, and UI toggles to restrict execution to NVLink-only shards.
- Per-model candle processes with dynamic port allocation and log streaming
- FastAPI backend with websocket push updates
- Vue 3 + Pinia frontend (same UI heritage as llama-cpp-studio)

## Smart Auto controls

The Smart Auto workflow now mirrors the full candle runtime surface area:

- **Speed vs quality slider** — continuously balances quantisation, KV cache and sampling knobs.
- **Use-case presets** — conversational, code, analysis and creative profiles tailor temperature, penalties and token budgets.
- **Concurrency hinting** — optional target concurrency and max token hints are honoured (and safely clamped to hardware limits).
- **Decision rationale** — enabling "Include decision rationale" on the UI surfaces backend reasoning (hardware snapshot, precision choice and per-parameter notes) before you apply changes.
- **Topology-aware controls** — toggle NVLink-only planning, inspect per-GPU budgets, and review backend recommendations for mixed NVLink/PCIe deployments.

A matching API surface is exposed via `POST /api/models/{model_id}/smart-auto`, accepting JSON payloads with the same fields used by the UI (`usage_mode`, `speed_quality`, `use_case`, `target_concurrency`, `max_tokens_hint`, `debug`).

## Getting Started

### 1. Requirements

- Rust toolchain (1.75+) with Cargo
- Python 3.10+
- Node.js 18+
- CUDA 12.x (Optional, for GPU builds)

### 2. Clone & Install

```bash
git clone https://github.com/your-org/candle-vllm-studio.git
cd candle-vllm-studio

# Backend dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Frontend dependencies
cd frontend
npm install
cd ..
```

### 3. Configure candle-vllm

The backend needs either a compiled candle binary or the source tree to call via `cargo run`:

```
export CANDLE_VLLM_BINARY=/path/to/candle-vllm/target/release/candle-vllm
# or
export CANDLE_VLLM_PATH=/path/to/candle-vllm
```

Optionally set a Hugging Face token:

```
export HUGGINGFACE_API_KEY=hf_xxx
```

### 4. Run

```bash
# Backend (FastAPI + uvicorn)
uvicorn backend.main:app --reload --port 8080

# Frontend
cd frontend
npm run dev
```

Open `http://localhost:5173` for the UI (or build the frontend and serve from the backend static path).

## Testing & Verification

- **Backend smoke test**: launch the backend with `uvicorn backend.main:app` and confirm the OpenAPI docs at `http://localhost:8080/docs`.
- **Smart auto sanity**: use the UI to generate configs for a downloaded model, ensure candle starts and the API is reachable at the displayed endpoint.
- **Frontend build**: `cd frontend && npm run build` (the built assets will be served automatically when placed under `frontend/dist`).
- **Unit tests**: `pytest backend/tests -q`
- **Smart Auto topology planning**: inspect the plan panel in the UI or call `POST /api/models/{id}/smart-auto` with `restrict_to_nvlink=true` to verify NVLink-aware budgeting.

Additional coverage (end-to-end runtime manager tests and UI automation) is still on the roadmap.

## Roadmap

- Extend unified monitoring coverage with richer GPU and runtime health metrics
- Expand Smart-Auto heuristics (documented constants live in `backend/smart_auto/constants.py`)
- Add CLI/Electron distribution packaging once the candle stack stabilises

## License

MIT (same as the original llama-cpp-studio). See `llama-cpp-studio/LICENSE` for details inherited from upstream.

