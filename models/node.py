"""
Node representation for GPU Cluster Management System
"""

from typing import List, Dict
# This is simplified node represeation, possilbe extensions
# Placment: 1 job can require multiple GPUs that can exceed node's total GPU count
# To Do: splitting logic across multiple nodes
# There is 1 to 1 mapping between gpu_id and job_id
# To Do: adding MIG support
# To Do: adding locks to the allocation and deallocation of GPUs

class Node:
    """Represents a compute node with GPUs"""
    
    def __init__(self, node_id: str, num_gpu: int, cpu_cores: int = 16, memory_gb: int = 64):
        self.node_id = node_id
        self.num_gpu = num_gpu
        self.free_gpus = num_gpu
        self.cpu_cores = cpu_cores
        self.memory_gb = memory_gb
        self.allocated_jobs: List[str] = []
        self.gpu_allocations: Dict[int, str] = {}  # gpu_id -> job_id
        
        # Resource monitoring
        self.cpu_utilization = 0.0
        self.memory_usage = 0.0
        self.network_in = 0.0
        self.network_out = 0.0
    
    def can_allocate(self, num_gpu: int) -> bool:
        """Check if node can allocate requested GPUs"""
        return self.free_gpus >= num_gpu
    
    def alloc_gpus(self, job_id: str, num_gpu: int) -> List[int]:
        """Allocate GPUs to a job"""
        if not self.can_allocate(num_gpu):
            return []

        allocated_gpu_ids = []
        for i in range(self.num_gpu):
            if i not in self.gpu_allocations and len(allocated_gpu_ids) < num_gpu:
                self.gpu_allocations[i] = job_id
                allocated_gpu_ids.append(i)
        
        self.free_gpus -= num_gpu
        self.allocated_jobs.append(job_id)
        return allocated_gpu_ids
    
    def release_gpus(self, job_id: str) -> int:
        """Release GPUs allocated to a job"""
        released_count = 0
        gpu_ids_to_remove = []
        
        for gpu_id, allocated_job_id in self.gpu_allocations.items():
            if allocated_job_id == job_id:
                gpu_ids_to_remove.append(gpu_id)
                released_count += 1
        
        for gpu_id in gpu_ids_to_remove:
            del self.gpu_allocations[gpu_id]
        
        self.free_gpus += released_count
        if job_id in self.allocated_jobs:
            self.allocated_jobs.remove(job_id)
        
        return released_count
    
    @property
    def utilization(self) -> float:
        """Node utilization percentage"""
        return (self.num_gpu - self.free_gpus) / self.num_gpu * 100
    
    @property
    def is_idle(self) -> bool:
        """Check if node is idle"""
        return self.free_gpus == self.num_gpu
    
    @property
    def is_full(self) -> bool:
        """Check if node is full"""
        return self.free_gpus == 0 