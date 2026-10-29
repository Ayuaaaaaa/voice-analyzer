from typing import Literal

from pydantic import BaseModel, Field, model_validator

EmotionalTone = Literal["neutral", "satisfied", "frustrated", "upset", "distressed"]
Intensity = Literal["low", "medium", "high"]
Severity = Literal["none", "low", "medium", "high"]
AudioQuality = Literal["clear", "slightly_impaired", "severely_impaired"]

FIELDS = [
    "emotional_tone", "emotional_intensity", "background_noise_present", "background_noise_type",
    "background_noise_severity", "audio_quality", "speaker_overlap_present", "long_silence_present", "confidence",
]


class AnalysisResult(BaseModel):
    emotional_tone: EmotionalTone
    emotional_intensity: Intensity
    background_noise_present: bool
    background_noise_type: str
    background_noise_severity: Severity
    audio_quality: AudioQuality
    speaker_overlap_present: bool
    long_silence_present: bool
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _noise_fields_consistent(self):
        if not self.background_noise_present:
            self.background_noise_type = ""
            self.background_noise_severity = "none"
        elif self.background_noise_severity == "none":
            self.background_noise_severity = "low"
        self.confidence = round(float(self.confidence), 2)
        return self
