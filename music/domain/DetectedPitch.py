from dataclasses import dataclass
from typing import Optional

from noteDetection.domain.DetectedNote import DetectedNote
from music.domain.MusicalNote import MusicalNote


@dataclass(frozen=True)
class DetectedPitch:
    detected: DetectedNote
    musical: Optional[MusicalNote]