"""Module 2b - policy-grounded severity classification."""

from aegisflow.severity.matrix import (
    LOW_CONFIDENCE_THRESHOLD,
    SeverityMatrix,
    describe_matrix,
)

__all__ = ["LOW_CONFIDENCE_THRESHOLD", "SeverityMatrix", "describe_matrix"]
