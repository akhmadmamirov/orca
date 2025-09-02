"""
Placement schemes package for GPU Cluster Management System
Contains various job placement strategies
"""

from .base import PlacementScheme
from .first_fit import FirstFitPlacement
from .best_fit import BestFitPlacement

__all__ = [
    'PlacementScheme',
    'FirstFitPlacement',
    'BestFitPlacement'
] 