# Safetensors Support Analysis - End-to-End Flow

## Current State (Post-Refactoring)

### ✅ What's Working
1. **Search & Discovery**
   - Backend can filter by model type (GGUF/safetensors/all)
   - Frontend has UI dropdown to select model type
   - Search results show safetensors metadata (file count, total size)
   - Extraction functions generalized to handle both formats

2. **Metadata Extraction**
   - `_extract_quantization()` returns "safetensors" for .safetensors files
   - `extract_base_model_name()` handles safetensors sharding (strips `-00001-of-00002`)
   - `extract_model_type()` uses both filename and HuggingFace ID for detection

3. **Download Logic**
   - Uses same download functions for both formats
   - Properly configures `weights_path` for safetensors (directory, not file)
   - Sets `weights_file=None` for safetensors to let candle-vllm find all shards

### ❌ What's Missing for Full Safetensors Support

## 1. **MULTI-FILE DOWNLOADS**

**Problem**: Safetensors models are often sharded into multiple files:
```
model-00001-of-00003.safetensors
model-00002-of-00003.safetensors
model-00003-of-00003.safetensors
config.json
tokenizer.json
tokenizer_config.json
```

**Current Behavior**: Users can only download ONE file at a time from the UI.

**Solution Needed**:
- Add a "Download All Safetensors Files" button in the frontend
- Create a new backend endpoint: `POST /api/models/download-safetensors-bundle`
- This endpoint should:
  1. Query HuggingFace API to get all `.safetensors` files for the repo
  2. Download all shards in parallel (with combined progress tracking)
  3. Also download required companion files: `config.json`, `tokenizer.json`, etc.
  4. Create a single database entry after all files are downloaded
  5. Handle partial failures gracefully

---

## 2. **FRONTEND UI IMPROVEMENTS**

**For Safetensors Models in Search Results**:

Currently shows:
```
✅ Safetensors Model
   Files: 3
   Total Size: 15.2 GB
   💡 Use --isq q4k for in-situ quantization
```

Should add:
```
📦 Download Options:
   [Download All Files (3 files, 15.2 GB)] ← NEW BUTTON
   
   Or download individual files:
   [Dropdown: model-00001-of-00003.safetensors]
   [Download Selected File]
```

---

## 3. **CONFIGURATION & RUNTIME SUPPORT**

**Current Config for Safetensors**:
```python
{
    "weights_path": "/data/models/Qwen/Qwen3-8B",  # Directory
    "weights_file": None,  # Let candle-vllm find all shards
    "isq": None,  # ← Should default to "q4k" for safetensors!
}
```

**Improvements Needed**:
- Smart Auto should recognize safetensors and auto-set `isq` if not already quantized
- UI should show a banner: "This is a safetensors model. Recommend using ISQ quantization."
- Config validation should check for incompatible settings (e.g., safetensors + GGUF-specific flags)

---

## 4. **SMART AUTO INTEGRATION**

Current Smart Auto logic is GGUF-centric. Need to add:

```python
def _choose_precision(self, model_profile, hardware_snapshot, slider):
    # ... existing GGUF logic ...
    
    # NEW: For safetensors models
    if model_profile.is_safetensors:
        if slider < 30:
            # Prioritize speed
            return PrecisionChoice(dtype="bf16", isq="q4k", ...)
        elif slider < 70:
            # Balanced
            return PrecisionChoice(dtype="bf16", isq="q5k", ...)
        else:
            # Prioritize quality
            return PrecisionChoice(dtype="bf16", isq="q6k", ...)
```

---

## 5. **MODEL PROFILE UPDATES**

`ModelProfile` (in `backend/smart_auto/model_metadata.py`) should include:

```python
@dataclass
class ModelProfile:
    # ... existing fields ...
    
    is_gguf: bool = False          # ✅ Already exists
    is_safetensors: bool = False   # ❌ NEW: Add this
    safetensors_shards: int = 0    # ❌ NEW: Number of shard files
    requires_isq: bool = False     # ❌ NEW: Flag for ISQ recommendation
```

---

## 6. **DATABASE SCHEMA**

Check if `Model` table supports safetensors properly:

```python
class Model(Base):
    # ... existing columns ...
    quantization = Column(String)  # ✅ Will be "safetensors"
    model_type = Column(String)    # ✅ Will be "qwen", "llama", etc.
    file_path = Column(String)     # ✅ For safetensors, this is directory path
    
    # ❌ NEW: Might need to add:
    # file_format = Column(String)  # "gguf" or "safetensors"
    # shard_count = Column(Integer)  # Number of safetensors shards
```

---

## 7. **CANDLE-VLLM COMMAND BUILDING**

In `backend/candle_manager.py`, `_build_command()`:

```python
def _build_command(self, config: Dict[str, Any]):
    # ... existing logic ...
    
    # For safetensors models
    if config.get("quantization") == "safetensors":
        # Use --w for directory (not --f for file)
        if weights_path:
            cmd.extend(["--w", str(weights_path)])
        
        # Recommend ISQ if not already set
        if not config.get("isq"):
            logger.warning("Safetensors model without ISQ. Recommend setting --isq q4k")
        
        # ISQ flag
        if config.get("isq"):
            cmd.extend(["--isq", config["isq"]])
```

---

## 8. **DOWNLOAD PROGRESS TRACKING**

For multi-file downloads, need enhanced progress:

```javascript
// Current (single file):
{
  "task_id": "download_xyz",
  "progress": 45,
  "bytes_downloaded": 4500000000,
  "total_bytes": 10000000000
}

// NEW (multi-file bundle):
{
  "task_id": "download_xyz",
  "bundle": true,
  "files": [
    {"name": "model-00001.safetensors", "progress": 100, "size": 5000000000},
    {"name": "model-00002.safetensors", "progress": 45, "size": 5000000000},
    {"name": "model-00003.safetensors", "progress": 0, "size": 5000000000}
  ],
  "overall_progress": 48,
  "total_size": 15000000000
}
```

---

## 9. **VALIDATION & ERROR HANDLING**

Add checks for:
- ✅ Safetensors models should not be used with `--f` flag
- ✅ Safetensors models should not use multi-GPU without checking candle-vllm support
- ❌ NEW: Warn if only partial shards downloaded
- ❌ NEW: Verify all required files present before allowing model start
- ❌ NEW: Check if `config.json` exists (required for safetensors)

---

## 10. **TESTING REQUIREMENTS**

Need to test:
1. Search for safetensors models ✅
2. Download single safetensors shard ✅ (works now)
3. Download complete safetensors bundle ❌ (not implemented)
4. Start safetensors model with ISQ ❌ (not tested)
5. Chat completion with safetensors model ❌ (not tested)
6. Multi-GPU with safetensors + ISQ ❌ (not tested)

---

## Implementation Priority

### 🔴 P0 - Critical (Required for Basic Safetensors Support)
1. Multi-file download endpoint (`download-safetensors-bundle`)
2. Frontend "Download All Files" button
3. Combined progress tracking for bundle downloads
4. ISQ recommendation in Smart Auto for safetensors

### 🟡 P1 - Important (Better UX)
1. Model Profile updates (`is_safetensors`, etc.)
2. Config validation for safetensors-specific settings
3. UI warnings/hints for ISQ usage
4. Partial download detection & recovery

### 🟢 P2 - Nice to Have
1. Database schema enhancements (shard_count, file_format)
2. Advanced progress tracking (per-file in bundle)
3. Automatic companion file downloads (tokenizer, config)
4. Smart Auto precision tuning for safetensors

---

## Next Steps

1. Implement P0 items (multi-file download)
2. Test with real safetensors model (e.g., `Qwen/Qwen3-8B`)
3. Verify chat completions work with safetensors + ISQ
4. Document safetensors workflow in README

