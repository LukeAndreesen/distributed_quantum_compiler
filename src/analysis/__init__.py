"""
src.analysis

Analysis and validation utilities.

Tools for validating partition candidates and computing basic cut metrics.

Public API:
- confirm_candidate_validity: Validate QPU capacity constraints per layer.
"""

from .cut_metrics import confirm_candidate_validity

__all__ = ["confirm_candidate_validity"]
