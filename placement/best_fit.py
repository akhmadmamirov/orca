"""
Best-fit placement scheme implementation
"""

from typing import List, Optional, Tuple
from models.job import Job
from models.node import Node
from .base import PlacementScheme


class BestFitPlacement(PlacementScheme):
    """Best-fit placement scheme (minimize fragmentation)"""
    
    def __init__(self):
        super().__init__("BestFit")
    
    def place_job(self, job: Job, nodes: List[Node]) -> Optional[Tuple[Node, List[int]]]:
        best_node = None
        best_fragmentation = float('inf')
        
        for node in nodes:
            if node.can_allocate(job.num_gpu):
                # Calculate fragmentation (unused GPUs after allocation)
                remaining_gpus = node.free_gpus - job.num_gpu
                if remaining_gpus < best_fragmentation:
                    best_fragmentation = remaining_gpus
                    best_node = node
        
        if best_node:
            allocated_gpus = best_node.alloc_gpus(job.job_id, job.num_gpu)
            if allocated_gpus:
                return best_node, allocated_gpus
        
        return None 