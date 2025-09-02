"""
Hybrid Priority Scheduler (HPS) implementation
A smart scheduler that combines multiple strategies without overcomplicating things
"""

import time
from typing import List, Optional, Dict
from models.job import Job
from .base import Scheduler


class HybridPriorityScheduler(Scheduler):
    """
    Hybrid Priority Scheduler that intelligently combines:
    - Shortest Job First (for efficiency)
    - Aging-based priority (for fairness)
    - Resource blocking prevention (for throughput)
    """
    
    def __init__(self, 
                 aging_threshold: float = 300.0,  # 5 minutes
                 aging_boost: float = 2.0,        # 2x priority boost
                 max_wait_time: float = 1800.0):  # 30 minutes max wait
        super().__init__("Hybrid-Priority")
        self.aging_threshold = aging_threshold
        self.aging_boost = aging_boost
        self.max_wait_time = max_wait_time
    
    def select_job(self, pending_jobs: List[Job]) -> None:
        if not pending_jobs:
            return None
        
        # Calculate priority scores for all jobs
        job_scores = []
        current_time = time.time()
        
        for job in pending_jobs:
            # Base score: shorter jobs get higher priority
            base_score = 1.0 / (1.0 + job.remaining_time / 3600)  # Normalize to hours
            
            # Aging boost: jobs waiting longer get priority
            wait_time = current_time - job.submit_time
            if wait_time > self.aging_threshold:
                # Calculate how much boost to give based on wait time
                wait_factor = min(wait_time / self.max_wait_time, 1.0)
                aging_score = self.aging_boost * wait_factor
            else:
                aging_score = 1.0
            
            # Resource blocking penalty: jobs using many GPUs get slight penalty
            # This prevents one large job from blocking many small ones
            gpu_penalty = 1.0 / (1.0 + job.num_gpu / 4)  # Normalize to 4 GPUs
            
            # Final score combines all factors
            final_score = base_score * aging_score * gpu_penalty
            
            job_scores.append((job, final_score))
        
        # Return job with highest score
        return max(job_scores, key=lambda x: x[1])[0]
    
    def get_scheduler_info(self) -> Dict[str, float]:
        """Get current scheduler configuration for monitoring"""
        return {
            "aging_threshold": self.aging_threshold,
            "aging_boost": self.aging_boost,
            "max_wait_time": self.max_wait_time
        } 