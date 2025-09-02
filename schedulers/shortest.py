"""
Shortest remaining time first scheduler implementation
"""

from typing import List, Optional
from models.job import Job
from .base import Scheduler


class ShortestScheduler(Scheduler):
    """Shortest remaining time first scheduler"""
    
    def __init__(self):
        super().__init__("Shortest")
    
    def select_job(self, pending_jobs: List[Job]) -> Optional[Job]:
        if not pending_jobs:
            return None
        # Return the job with shortest remaining time
        return min(pending_jobs, key=lambda j: j.remaining_time) 