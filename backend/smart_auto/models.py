"""Minimal data models used by the smart_auto helpers."""
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ModelMetadata:
    """Subset of GGUF-derived metadata required by smart_auto."""
    layer_count: int
    architecture: str
    context_length: int
    vocab_size: int
    embedding_length: int
    attention_head_count: int
    attention_head_count_kv: int
    block_count: int = 0
    is_moe: bool = False
    expert_count: int = 0
    experts_used_count: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        return cls(
            layer_count=data.get("layer_count", 32),
            architecture=data.get("architecture", "unknown"),
            context_length=data.get("context_length", 0),
            vocab_size=data.get("vocab_size", 0),
            embedding_length=data.get("embedding_length", 0),
            attention_head_count=data.get("attention_head_count", 0),
            attention_head_count_kv=data.get("attention_head_count_kv", 0),
            block_count=data.get("block_count", 0),
            is_moe=data.get("is_moe", False),
            expert_count=data.get("expert_count", 0),
            experts_used_count=data.get("experts_used_count", 0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_count": self.layer_count,
            "architecture": self.architecture,
            "context_length": self.context_length,
            "vocab_size": self.vocab_size,
            "embedding_length": self.embedding_length,
            "attention_head_count": self.attention_head_count,
            "attention_head_count_kv": self.attention_head_count_kv,
            "block_count": self.block_count,
            "is_moe": self.is_moe,
            "expert_count": self.expert_count,
            "experts_used_count": self.experts_used_count,
        }

