"""
Base placement scheme class for GPU Cluster Management System
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from models.job import Job
from models.node import Node


class PlacementScheme(ABC):
    """Base placement scheme class"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def place_job(self, job: Job, nodes: List[Node]) -> Optional[Tuple[Node, List[int]]]:
        """Place job on nodes - to be implemented by subclasses"""
        pass 