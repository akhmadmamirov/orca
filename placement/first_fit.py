"""
First-fit placement scheme implementation
"""

from typing import List, Optional, Tuple
from models.job import Job
from models.node import Node
from .base import PlacementScheme


class FirstFitPlacement(PlacementScheme):
    """First-fit placement scheme"""
    
    def __init__(self):
        super().__init__("FirstFit")
    
    def place_job(self, job: Job, nodes: List[Node]) -> Optional[Tuple[Node, List[int]]]:
        for node in nodes:
            if node.can_allocate(job.num_gpu):
                allocated_gpus = node.alloc_gpus(job.job_id, job.num_gpu)
                if allocated_gpus:
                    return node, allocated_gpus
        return None 