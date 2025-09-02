"""
Main GPU cluster management system
"""

from typing import List, Dict
from models.job import Job, JobState
from models.node import Node
from schedulers import FIFOScheduler, SJFScheduler, ShortestScheduler, ShortestGPUScheduler
from placement import FirstFitPlacement, BestFitPlacement
from metrics import MetricsCollector


class GPUClusterManager:
    """Main GPU cluster management system"""
    
    def __init__(self, num_nodes: int = 4, gpus_per_node: int = 4):
        self.nodes = [f"node_{i}" for i in range(num_nodes)]
        self.node_objects = {node_id: Node(node_id, gpus_per_node) for node_id in self.nodes}
        
        # Job management
        self.pending_jobs: List[Job] = []
        self.running_jobs: List[Job] = []
        self.completed_jobs: List[Job] = []
        
        # Scheduling and placement
        self.schedulers = {
            'fifo': FIFOScheduler(),
            'sjf': SJFScheduler(),
            'shortest': ShortestScheduler(),
            'shortest-gpu': ShortestGPUScheduler()
        }
        self.current_scheduler = 'fifo'
        
        self.placement_schemes = {
            'first-fit': FirstFitPlacement(),
            'best-fit': BestFitPlacement()
        }
        self.current_placement = 'first-fit'
        
        # Metrics
        self.metrics = MetricsCollector()
        self.simulation_time = 0.0
        
        # Job ID counter
        self.job_counter = 0
    
    def submit_job(self, num_gpu: int, iterations: int, model_name: str, 
                   duration: float, interval: float = 1.0, submit_time: float = None) -> str:
        """Submit a new job to the system"""
        self.job_counter += 1
        job_id = f"job_{self.job_counter}"
        
        # Use provided submit_time or current simulation time
        actual_submit_time = submit_time if submit_time is not None else self.simulation_time
        
        job = Job(
            job_id=job_id,
            num_gpu=num_gpu,
            submit_time=actual_submit_time,
            iterations=iterations,
            model_name=model_name,
            duration=duration,
            interval=interval
        )
        
        job.state = JobState.PENDING
        self.pending_jobs.append(job)
        
        print(f"Submitted {job_id}: {num_gpu} GPUs, {model_name}, duration: {duration}")
        return job_id
    
    def schedule_jobs(self):
        """Schedule pending jobs using current scheduler"""
        scheduler = self.schedulers[self.current_scheduler]
        placement = self.placement_schemes[self.current_placement]
        
        while self.pending_jobs:
            job = scheduler.select_job(self.pending_jobs)
            if not job:
                break
            
            # Try to place the job
            placement_result = placement.place_job(job, list(self.node_objects.values()))
            
            if placement_result:
                node, allocated_gpus = placement_result
                self._start_job(job, node, allocated_gpus)
                self.pending_jobs.remove(job)
            else:
                # Job cannot be placed, keep in pending
                break
    
    def _start_job(self, job: Job, node: Node, allocated_gpus: List[int]):
        """Start a job on a node"""
        job.state = JobState.RUNNING
        job.start_time = self.simulation_time
        job.allocated_gpus = [f"{node.node_id}_gpu_{gpu_id}" for gpu_id in allocated_gpus]
        self.running_jobs.append(job)
        
        print(f"Started {job.job_id} on {node.node_id} with GPUs: {allocated_gpus}")
    
    def _complete_job(self, job: Job):
        """Complete a running job"""
        job.state = JobState.END
        job.end_time = self.simulation_time
        job.execution_time = job.end_time - job.start_time if job.start_time else 0
        
        # Release GPUs
        for node in self.node_objects.values():
            if job.job_id in node.allocated_jobs:
                node.release_gpus(job.job_id)
                break
        
        # Move to completed
        self.running_jobs.remove(job)
        self.completed_jobs.append(job)
        
        # Record metrics
        completion_time = job.total_time
        self.metrics.record_job_metric(job.job_id, 'completion_time', completion_time)
        self.metrics.record_job_metric(job.job_id, 'execution_time', job.execution_time)
        self.metrics.record_job_metric(job.job_id, 'pending_time', job.pending_time)
        
        print(f"Completed {job.job_id} in {completion_time:.2f}s")
    
    def update_simulation(self, time_step: float = 1.0):
        """Update simulation state"""
        self.simulation_time += time_step
        
        # Update running jobs
        for job in self.running_jobs[:]:  # Copy list to avoid modification during iteration
            job.execution_time += time_step
            
            # Check if job is complete
            if job.execution_time >= job.duration:
                self._complete_job(job)
        
        # Update pending jobs
        for job in self.pending_jobs:
            job.pending_time += time_step
        
        # Update resource metrics
        self._update_resource_metrics()
        
        # Try to schedule new jobs
        self.schedule_jobs()
    
    def _update_resource_metrics(self):
        """Update resource utilization metrics"""
        total_gpus = sum(node.num_gpu for node in self.node_objects.values())
        used_gpus = sum(node.num_gpu - node.free_gpus for node in self.node_objects.values())
        
        gpu_utilization = (used_gpus / total_gpus) * 100 if total_gpus > 0 else 0
        self.metrics.record_cluster_metric('gpu_utilization', gpu_utilization)
        
        # Calculate resource fragmentation
        fragmentation = 0
        for node in self.node_objects.values():
            if node.free_gpus > 0 and node.free_gpus < node.num_gpu:
                fragmentation += node.free_gpus / node.num_gpu
        
        self.metrics.record_cluster_metric('resource_fragmentation', fragmentation)
        
        # Record per-node metrics
        for node_id, node in self.node_objects.items():
            self.metrics.record_resource_metric(node_id, 'gpu_utilization', node.utilization)
            self.metrics.record_resource_metric(node_id, 'cpu_utilization', node.cpu_utilization)
            self.metrics.record_resource_metric(node_id, 'memory_usage', node.memory_usage)
    
    def set_scheduler(self, scheduler_name: str):
        """Change the active scheduler"""
        if scheduler_name in self.schedulers:
            self.current_scheduler = scheduler_name
            print(f"Switched to {scheduler_name} scheduler")
        else:
            print(f"Unknown scheduler: {scheduler_name}")
    
    def set_placement(self, placement_name: str):
        """Change the active placement scheme"""
        if placement_name in self.placement_schemes:
            self.current_placement = placement_name
            print(f"Switched to {placement_name} placement")
        else:
            print(f"Unknown placement: {placement_name}")
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            'simulation_time': self.simulation_time,
            'pending_jobs': len(self.pending_jobs),
            'running_jobs': len(self.running_jobs),
            'completed_jobs': len(self.completed_jobs),
            'total_nodes': len(self.nodes),
            'gpu_utilization': self.metrics.get_gpu_utilization(),
            'average_jct': self.metrics.get_average_jct(),
            'resource_fragmentation': self.metrics.get_resource_fragmentation()
        }
    
    def print_status(self):
        """Print current system status"""
        status = self.get_system_status()
        print("\n" + "="*50)
        print("GPU CLUSTER STATUS")
        print("="*50)
        print(f"Time: {status['simulation_time']:.1f}s")
        print(f"Jobs: {status['pending_jobs']} pending, {status['running_jobs']} running, {status['completed_jobs']} completed")
        print(f"GPU Utilization: {status['gpu_utilization']:.1f}%")
        print(f"Average JCT: {status['average_jct']:.2f}s")
        print(f"Resource Fragmentation: {status['resource_fragmentation']:.2f}")
        print(f"Current Scheduler: {self.current_scheduler}")
        print(f"Current Placement: {self.current_placement}")
        
        print("\nNode Status:")
        for node_id, node in self.node_objects.items():
            print(f"  {node_id}: {node.num_gpu - node.free_gpus}/{node.num_gpu} GPUs used ({node.utilization:.1f}%)")
        
        print("="*50) 