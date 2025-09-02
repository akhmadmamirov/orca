"""
Smallest job first scheduler implementation
"""

from typing import List, Optional
from models.job import Job
from .base import Scheduler


class SJFScheduler(Scheduler):
    """Smallest job first scheduler (by GPU count)"""
    
    def __init__(self):
        super().__init__("SJF")
    
    def select_job(self, pending_jobs: List[Job]) -> Optional[Job]:
        if not pending_jobs:
            return None
        # Return the job with smallest GPU requirement
        return min(pending_jobs, key=lambda j: j.num_gpu) 