"""
Job representation for GPU Cluster Management System
"""

import time
from enum import Enum
from typing import List, Optional
from dataclasses import dataclass


class JobState(Enum):
    """Job states in the system"""
    ADDED = "ADDED"
    EVENT = "EVENT"
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    END = "END"
    ERROR = "ERROR"


@dataclass
class Job:
    """Represents a deep learning job"""
    job_id: str
    num_gpu: int
    submit_time: float
    iterations: int
    model_name: str
    duration: float
    interval: float
    
    # Runtime fields
    state: JobState = JobState.ADDED
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    execution_time: float = 0.0
    pending_time: float = 0.0
    preemption_count: int = 0
    resume_count: int = 0
    allocated_gpus: List[str] = None
    
    def __post_init__(self):
        if self.allocated_gpus is None:
            self.allocated_gpus = []
    
    @property
    def total_time(self) -> float:
        """Total time from submit to completion"""
        if self.end_time and self.start_time:
            return self.end_time - self.submit_time
        return time.time() - self.submit_time
    
    @property
    def remaining_time(self) -> float:
        """Estimated remaining execution time"""
        if self.state == JobState.RUNNING:
            return max(0, self.duration - self.execution_time)
        return self.duration 