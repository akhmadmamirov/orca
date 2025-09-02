"""
Predictive Backfill Scheduler (PBS) implementation
A scheduler that looks ahead to find optimal job combinations and fills gaps efficiently
"""

import time
from typing import List, Optional, Dict, Tuple
from models.job import Job
from .base import Scheduler


class PredictiveBackfillScheduler(Scheduler):
    """
    Predictive Backfill Scheduler that:
    - Looks ahead to find optimal job combinations
    - Fills GPU gaps efficiently
    - Prevents fragmentation
    - Maximizes resource utilization
    """
    
    def __init__(self, 
                 lookahead_jobs: int = 5,      # How many jobs to look ahead
                 min_gpu_threshold: int = 2,    # Minimum GPUs to consider for backfill
                 time_window: float = 3600.0):  # 1 hour lookahead window
        super().__init__("Predictive-Backfill")
        self.lookahead_jobs = lookahead_jobs
        self.min_gpu_threshold = min_gpu_threshold
        self.time_window = time_window
    
    def select_job(self, pending_jobs: List[Job]) -> Optional[Job]:
        if not pending_jobs:
            return None
        
        # Sort jobs by different criteria to find best combinations
        jobs_by_duration = sorted(pending_jobs, key=lambda j: j.remaining_time)
        jobs_by_gpu = sorted(pending_jobs, key=lambda j: j.num_gpu)
        jobs_by_efficiency = sorted(pending_jobs, 
                                  key=lambda j: j.iterations / (j.num_gpu * j.remaining_time), 
                                  reverse=True)
        
        # Strategy 1: Find the most efficient job (highest work per GPU per time)
        if jobs_by_efficiency:
            best_efficient = jobs_by_efficiency[0]
            efficiency_score = best_efficient.iterations / (best_efficient.num_gpu * best_efficient.remaining_time)
            
            # If efficiency is significantly high, prioritize it
            if efficiency_score > 0.1:  # Threshold for "efficient enough"
                return best_efficient
        
        # Strategy 2: Look for small jobs that can fit in gaps
        small_jobs = [j for j in pending_jobs if j.num_gpu <= self.min_gpu_threshold]
        if small_jobs:
            # Pick the shortest small job to clear the queue faster
            return min(small_jobs, key=lambda j: j.remaining_time)
        
        # Strategy 3: Find jobs that won't block others for too long
        medium_jobs = [j for j in pending_jobs if j.remaining_time < self.time_window]
        if medium_jobs:
            # Pick the one with fewest GPUs to minimize blocking
            return min(medium_jobs, key=lambda j: j.num_gpu)
        
        # Strategy 4: Default to shortest remaining time
        return min(pending_jobs, key=lambda j: j.remaining_time)
    
    def find_optimal_combination(self, pending_jobs: List[Job]) -> List[Job]:
        """
        Find optimal combination of jobs that can run together efficiently.
        This is the key innovation - looking ahead to find synergies.
        """
        if len(pending_jobs) < 2:
            return pending_jobs[:1] if pending_jobs else []
        
        # Group jobs by GPU requirements
        gpu_groups = {}
        for job in pending_jobs:
            gpu_count = job.num_gpu
            if gpu_count not in gpu_groups:
                gpu_groups[gpu_count] = []
            gpu_groups[gpu_count].append(job)
        
        # Find complementary job pairs
        optimal_pairs = []
        for gpu1 in gpu_groups:
            for gpu2 in gpu_groups:
                if gpu1 + gpu2 <= 8:  # Assuming 8 GPU system
                    # Find best pair
                    for job1 in gpu_groups[gpu1]:
                        for job2 in gpu_groups[gpu2]:
                            if job1 != job2:
                                # Calculate combined efficiency
                                total_work = job1.iterations + job2.iterations
                                total_gpu_time = (job1.num_gpu * job1.remaining_time + 
                                                job2.num_gpu * job2.remaining_time)
                                combined_efficiency = total_work / total_gpu_time
                                
                                optimal_pairs.append(([job1, job2], combined_efficiency))
        
        if optimal_pairs:
            # Return the most efficient combination
            best_pair = max(optimal_pairs, key=lambda x: x[1])
            return best_pair[0]
        
        return [pending_jobs[0]]  # Fallback to first job
    
    def get_scheduler_info(self) -> Dict[str, any]:
        """Get current scheduler configuration for monitoring"""
        return {
            "lookahead_jobs": self.lookahead_jobs,
            "min_gpu_threshold": self.min_gpu_threshold,
            "time_window": self.time_window,
            "strategy": "predictive_backfill"
        } 