"""Agent implementations for test runner."""

from .simple_agent import SimpleAutonomousAgent
from .langcoop_agent import LangCoopAgent

try:
	from .langcoop_vlm_agent import LangCoopVLMAgent
except ModuleNotFoundError:
	LangCoopVLMAgent = None

__all__ = ["SimpleAutonomousAgent", "LangCoopAgent"]
if LangCoopVLMAgent is not None:
	__all__.append("LangCoopVLMAgent")
