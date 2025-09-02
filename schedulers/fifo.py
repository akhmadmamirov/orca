"""
First-in-first-out scheduler implementation
"""

from typing import List, Optional
from models.job import Job
from .base import Scheduler


class FIFOScheduler(Scheduler):
    """First-in-first-out scheduler"""
    
    def __init__(self):
        super().__init__("FIFO")
    
    def select_job(self, pending_jobs: List[Job]) -> Optional[Job]:
        if not pending_jobs:
            return None
        # Return the job with earliest submit time
        return min(pending_jobs, key=lambda j: j.submit_time) 