"""Usage profiles and heuristics constants."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple


BandValues = Mapping[str, float]


@dataclass(frozen=True)
class SamplingWindow:
    """Three anchor points for UI slider interpolation."""

    speed: float
    balanced: float
    quality: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "speed": float(self.speed),
            "balanced": float(self.balanced),
            "quality": float(self.quality),
        }


@dataclass(frozen=True)
class IntegerWindow:
    speed: int
    balanced: int
    quality: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "speed": float(self.speed),
            "balanced": float(self.balanced),
            "quality": float(self.quality),
        }


@dataclass(frozen=True)
class UsageProfile:
    key: str
    label: str
    temperature: SamplingWindow
    top_p: SamplingWindow
    top_k: IntegerWindow
    min_p: SamplingWindow
    frequency_penalty: SamplingWindow
    presence_penalty: SamplingWindow
    generation_ratio: SamplingWindow  # fraction of context length
    min_generation: int
    concurrency_single_user: IntegerWindow
    concurrency_multi_user: IntegerWindow
    block_size: IntegerWindow
    prefill_chunk_size: IntegerWindow
    quality_bias: float  # Allows per-use-case skew of slider (0 neutral)

    def sampling_windows(self) -> Dict[str, BandValues]:
        return {
            "temperature": self.temperature.as_dict(),
            "top_p": self.top_p.as_dict(),
            "top_k": self.top_k.as_dict(),
            "min_p": self.min_p.as_dict(),
            "frequency_penalty": self.frequency_penalty.as_dict(),
            "presence_penalty": self.presence_penalty.as_dict(),
        }


USAGE_PROFILES: Dict[str, UsageProfile] = {
    "chat": UsageProfile(
        key="chat",
        label="Conversational",
        # Temperature: Higher=creative, Lower=focused. Chat benefits from moderate creativity
        temperature=SamplingWindow(speed=0.8, balanced=0.7, quality=0.6),
        # Top-p: Nucleus sampling. Higher=more diverse. Chat needs balance
        top_p=SamplingWindow(speed=0.92, balanced=0.9, quality=0.88),
        # Top-k: Limits vocabulary. Lower=more focused/coherent
        top_k=IntegerWindow(speed=50, balanced=40, quality=30),
        # Min-p: Minimum probability threshold. Prevents low-quality tokens
        min_p=SamplingWindow(speed=0.05, balanced=0.08, quality=0.12),
        # Repetition penalties: Should be minimal for natural chat
        frequency_penalty=SamplingWindow(speed=0.0, balanced=0.0, quality=0.0),
        presence_penalty=SamplingWindow(speed=0.0, balanced=0.0, quality=0.0),
        generation_ratio=SamplingWindow(speed=0.18, balanced=0.22, quality=0.28),
        min_generation=512,
        concurrency_single_user=IntegerWindow(speed=2, balanced=1, quality=1),
        concurrency_multi_user=IntegerWindow(speed=6, balanced=4, quality=2),
        block_size=IntegerWindow(speed=128, balanced=256, quality=320),
        prefill_chunk_size=IntegerWindow(speed=1536, balanced=1024, quality=768),
        quality_bias=0.0,
    ),
    "code": UsageProfile(
        key="code",
        label="Code generation",
        # Code needs low temperature for deterministic, correct output
        temperature=SamplingWindow(speed=0.3, balanced=0.2, quality=0.1),
        # Lower top-p for code to prioritize correct syntax
        top_p=SamplingWindow(speed=0.85, balanced=0.8, quality=0.75),
        # Higher top-k for code vocabulary diversity
        top_k=IntegerWindow(speed=60, balanced=50, quality=40),
        # Higher min-p to filter out unlikely/incorrect tokens
        min_p=SamplingWindow(speed=0.05, balanced=0.08, quality=0.1),
        # No penalties - let the model use standard patterns
        frequency_penalty=SamplingWindow(speed=0.0, balanced=0.0, quality=0.0),
        presence_penalty=SamplingWindow(speed=0.0, balanced=0.0, quality=0.0),
        generation_ratio=SamplingWindow(speed=0.12, balanced=0.16, quality=0.2),
        min_generation=256,
        concurrency_single_user=IntegerWindow(speed=2, balanced=1, quality=1),
        concurrency_multi_user=IntegerWindow(speed=4, balanced=3, quality=2),
        block_size=IntegerWindow(speed=128, balanced=192, quality=256),
        prefill_chunk_size=IntegerWindow(speed=1024, balanced=768, quality=512),
        quality_bias=0.1,
    ),
    "analysis": UsageProfile(
        key="analysis",
        label="Analysis / long-form",
        # Moderate temperature for coherent analysis
        temperature=SamplingWindow(speed=0.6, balanced=0.5, quality=0.4),
        # High top-p for nuanced vocabulary
        top_p=SamplingWindow(speed=0.92, balanced=0.9, quality=0.88),
        # Moderate top-k for professional language
        top_k=IntegerWindow(speed=50, balanced=40, quality=35),
        # Higher min-p for quality filtering
        min_p=SamplingWindow(speed=0.08, balanced=0.1, quality=0.12),
        # Light penalties to reduce repetition in long-form content
        frequency_penalty=SamplingWindow(speed=0.1, balanced=0.15, quality=0.2),
        presence_penalty=SamplingWindow(speed=0.05, balanced=0.08, quality=0.1),
        generation_ratio=SamplingWindow(speed=0.25, balanced=0.32, quality=0.4),
        min_generation=1024,
        concurrency_single_user=IntegerWindow(speed=1, balanced=1, quality=1),
        concurrency_multi_user=IntegerWindow(speed=3, balanced=2, quality=2),
        block_size=IntegerWindow(speed=256, balanced=320, quality=384),
        prefill_chunk_size=IntegerWindow(speed=2048, balanced=1536, quality=1024),
        quality_bias=-0.05,
    ),
    "creative": UsageProfile(
        key="creative",
        label="Creative writing",
        # Higher temperature for creativity, but not too high to avoid gibberish
        temperature=SamplingWindow(speed=0.9, balanced=0.8, quality=0.7),
        # Very high top-p for maximum vocabulary diversity
        top_p=SamplingWindow(speed=0.95, balanced=0.92, quality=0.9),
        # Higher top-k for creative word choice
        top_k=IntegerWindow(speed=80, balanced=60, quality=45),
        # Lower min-p to allow more creative/unusual tokens
        min_p=SamplingWindow(speed=0.03, balanced=0.05, quality=0.06),
        # Moderate penalties to encourage variety without breaking flow
        frequency_penalty=SamplingWindow(speed=0.3, balanced=0.4, quality=0.5),
        presence_penalty=SamplingWindow(speed=0.2, balanced=0.3, quality=0.4),
        generation_ratio=SamplingWindow(speed=0.2, balanced=0.3, quality=0.38),
        min_generation=768,
        concurrency_single_user=IntegerWindow(speed=2, balanced=1, quality=1),
        concurrency_multi_user=IntegerWindow(speed=5, balanced=3, quality=2),
        block_size=IntegerWindow(speed=192, balanced=256, quality=320),
        prefill_chunk_size=IntegerWindow(speed=1792, balanced=1280, quality=896),
        quality_bias=-0.08,
    ),
}


def resolve_usage_profile(use_case: str | None) -> UsageProfile:
    if not use_case:
        return USAGE_PROFILES["chat"]
    use_case = use_case.lower()
    return USAGE_PROFILES.get(use_case, USAGE_PROFILES["chat"])

