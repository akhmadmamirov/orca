"""
Shortest remaining GPU time first scheduler implementation
"""

from typing import List, Optional
from models.job import Job
from .base import Scheduler


class ShortestGPUScheduler(Scheduler):
    """Shortest remaining GPU time first scheduler"""
    
    def __init__(self):
        super().__init__("Shortest-GPU")
    
    def select_job(self, pending_jobs: List[Job]) -> Optional[Job]:
        if not pending_jobs:
            return None
        # Return the job with shortest remaining GPU time (duration * num_gpu)
        return min(pending_jobs, key=lambda j: j.remaining_time * j.num_gpu) 