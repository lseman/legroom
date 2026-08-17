"""Output module — output shaping and verbosity control."""

from .policy import OutputPolicy
from .shaper import OutputShaper
from .turn_classify import classify_turn

__all__ = ["OutputPolicy", "OutputShaper", "classify_turn"]
