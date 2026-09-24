"""TrajLens: deterministic auditing of AI-agent tool-use trajectories."""
from .verifiers import CRITERIA, VERIFIERS, CriterionResult, grade

__version__ = "3.0.0"
__all__ = ["CRITERIA", "VERIFIERS", "CriterionResult", "grade", "__version__"]
