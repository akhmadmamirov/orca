"""
Smart Batch Scheduler (SBS) implementation
A scheduler that groups similar jobs together for optimal batch execution
"""

import time
from typing import List, Optional, Dict, Tuple
from models.job import Job
from .base import Scheduler


class SmartBatchScheduler(Scheduler):
    """
    Smart Batch Scheduler that:
    - Groups similar jobs together for batch processing
    - Optimizes GPU utilization through intelligent batching
    - Reduces context switching overhead
    - Maximizes throughput through job synergies
    """
    
    def __init__(self, 
                 batch_size_threshold: int = 3,     # Minimum jobs to form a batch
                 similarity_threshold: float = 0.8,  # How similar jobs must be
                 max_batch_gpu: int = 8):           # Maximum GPUs per batch
        super().__init__("Smart-Batch")
        self.batch_size_threshold = batch_size_threshold
        self.similarity_threshold = similarity_threshold
        self.max_batch_gpu = max_batch_gpu
    
    def select_job(self, pending_jobs: List[Job]) -> Optional[Job]:
        if not pending_jobs:
            return None
        
        # Try to find optimal batch first
        optimal_batch = self._find_optimal_batch(pending_jobs)
        if optimal_batch and len(optimal_batch) >= self.batch_size_threshold:
            # Return the first job from the optimal batch
            return optimal_batch[0]
        
        # If no good batch, fall back to smart individual selection
        return self._select_best_individual_job(pending_jobs)
    
    def _find_optimal_batch(self, pending_jobs: List[Job]) -> Optional[List[Job]]:
        """Find the best batch of similar jobs"""
        if len(pending_jobs) < self.batch_size_threshold:
            return None
        
        # Group jobs by model type (similar models can often share resources)
        model_groups = {}
        for job in pending_jobs:
            model_type = self._get_model_family(job.model_name)
            if model_type not in model_groups:
                model_groups[model_type] = []
            model_groups[model_type].append(job)
        
        best_batch = None
        best_score = 0
        
        # Evaluate each model group for batching potential
        for model_type, jobs in model_groups.items():
            if len(jobs) >= self.batch_size_threshold:
                # Try to find optimal subset
                for batch_size in range(self.batch_size_threshold, min(len(jobs) + 1, 6)):
                    # Try different combinations
                    for i in range(len(jobs) - batch_size + 1):
                        batch = jobs[i:i + batch_size]
                        batch_score = self._calculate_batch_score(batch)
                        
                        if batch_score > best_score:
                            best_score = batch_score
                            best_batch = batch
        
        return best_batch
    
    def _get_model_family(self, model_name: str) -> str:
        """Extract model family from model name"""
        model_name_lower = model_name.lower()
        
        if 'resnet' in model_name_lower:
            return 'resnet'
        elif 'bert' in model_name_lower:
            return 'bert'
        elif 'transformer' in model_name_lower:
            return 'transformer'
        elif 'lstm' in model_name_lower:
            return 'lstm'
        else:
            return 'other'
    
    def _calculate_batch_score(self, batch: List[Job]) -> float:
        """Calculate how good a batch is"""
        if not batch:
            return 0
        
        # Calculate total resources needed
        total_gpus = sum(job.num_gpu for job in batch)
        if total_gpus > self.max_batch_gpu:
            return 0  # Invalid batch
        
        # Calculate batch efficiency
        total_work = sum(job.iterations for job in batch)
        total_time = max(job.remaining_time for job in batch)  # Batch time is limited by longest job
        total_gpu_time = total_gpus * total_time
        
        # Efficiency = work per GPU per time
        efficiency = total_work / total_gpu_time if total_gpu_time > 0 else 0
        
        # Bonus for similar job characteristics
        duration_variance = self._calculate_variance([job.remaining_time for job in batch])
        gpu_variance = self._calculate_variance([job.num_gpu for job in batch])
        
        # Lower variance = better batch
        similarity_bonus = 1.0 / (1.0 + duration_variance + gpu_variance)
        
        return efficiency * similarity_bonus
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values"""
        if len(values) < 2:
            return 0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def _select_best_individual_job(self, pending_jobs: List[Job]) -> Optional[Job]:
        """Select the best individual job when batching isn't optimal"""
        if not pending_jobs:
            return None
        
        # Score jobs based on multiple factors
        job_scores = []
        
        for job in pending_jobs:
            # Base efficiency score
            efficiency = job.iterations / (job.num_gpu * job.remaining_time) if job.remaining_time > 0 else 0
            
            # GPU utilization score (prefer jobs that use GPUs efficiently)
            gpu_score = 1.0 / (1.0 + job.num_gpu / 4)  # Normalize to 4 GPUs
            
            # Time score (prefer shorter jobs)
            time_score = 1.0 / (1.0 + job.remaining_time / 3600)  # Normalize to hours
            
            # Combined score
            final_score = efficiency * gpu_score * time_score
            job_scores.append((job, final_score))
        
        # Return job with highest score
        return max(job_scores, key=lambda x: x[1])[0]
    
    def get_scheduler_info(self) -> Dict[str, any]:
        """Get current scheduler configuration for monitoring"""
        return {
            "batch_size_threshold": self.batch_size_threshold,
            "similarity_threshold": self.similarity_threshold,
            "max_batch_gpu": self.max_batch_gpu,
            "strategy": "smart_batching"
        } 