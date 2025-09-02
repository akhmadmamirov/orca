"""
Base scheduler class for GPU Cluster Management System
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from models.job import Job


class Scheduler(ABC):
    """Base scheduler class"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def select_job(self, pending_jobs: List[Job]) -> Optional[Job]:
        """Select next job to schedule - to be implemented by subclasses"""
        pass 