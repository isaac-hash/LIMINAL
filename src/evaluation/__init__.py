"""Evaluation metrics and analysis tools."""
from src.evaluation.metrics import (
    compute_accuracy,
    count_parameters,
    evaluate_model_detailed,
)

__all__ = [
    "compute_accuracy",
    "count_parameters",
    "evaluate_model_detailed",
]
