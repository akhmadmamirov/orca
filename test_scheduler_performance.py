#!/usr/bin/env python3
"""
Comprehensive Scheduler Performance Testing Framework
Tests all schedulers with 5k jobs and analyzes their performance
"""

import time
import random
import statistics
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.job import Job, JobState
from models.node import Node
from schedulers import (
    FIFOScheduler, SJFScheduler, ShortestScheduler, ShortestGPUScheduler,
    HybridPriorityScheduler, PredictiveBackfillScheduler, SmartBatchScheduler
)


@dataclass
class SchedulerResult:
    """Results from a single scheduler test"""
    scheduler_name: str
    total_jobs: int
    completed_jobs: int
    total_time: float
    avg_wait_time: float
    avg_execution_time: float
    gpu_utilization: float
    throughput: float  # jobs per hour
    fairness_score: float
    job_completion_times: List[float]
    wait_times: List[float]
    execution_times: List[float]


class ClusterSimulator:
    """Simulates a GPU cluster for testing schedulers"""
    
    def __init__(self, num_nodes: int = 4, gpus_per_node: int = 8):
        self.nodes = [Node(f"node_{i}", gpus_per_node) for i in range(num_nodes)]
        self.total_gpus = num_nodes * gpus_per_node
        self.running_jobs: Dict[str, Job] = {}
        self.completed_jobs: List[Job] = []
        self.current_time = 0.0
        
    def can_allocate_job(self, job: Job) -> bool:
        """Check if we can allocate resources for a job"""
        total_free_gpus = sum(node.free_gpus for node in self.nodes)
        return total_free_gpus >= job.num_gpu
    
    def allocate_job(self, job: Job) -> bool:
        """Allocate resources for a job"""
        if not self.can_allocate_job(job):
            return False
        
        # Find nodes with enough free GPUs
        allocated_gpus = []
        remaining_gpus_needed = job.num_gpu
        
        for node in self.nodes:
            if remaining_gpus_needed <= 0:
                break
            if node.free_gpus > 0:
                gpus_to_allocate = min(node.free_gpus, remaining_gpus_needed)
                gpu_ids = node.alloc_gpus(job.job_id, gpus_to_allocate)
                allocated_gpus.extend([f"{node.node_id}:{gpu_id}" for gpu_id in gpu_ids])
                remaining_gpus_needed -= gpus_to_allocate
        
        if len(allocated_gpus) == job.num_gpu:
            job.allocated_gpus = allocated_gpus
            job.state = JobState.RUNNING
            job.start_time = self.current_time
            self.running_jobs[job.job_id] = job
            return True
        
        return False
    
    def advance_time(self, time_step: float):
        """Advance simulation time and process running jobs"""
        self.current_time += time_step
        
        # Process running jobs
        completed_jobs = []
        for job_id, job in list(self.running_jobs.items()):
            job.execution_time += time_step
            
            if job.execution_time >= job.duration:
                # Job completed
                job.state = JobState.END
                job.end_time = self.current_time
                self.completed_jobs.append(job)
                completed_jobs.append(job_id)
                
                # Release GPUs
                for gpu_allocation in job.allocated_gpus:
                    node_id, gpu_id = gpu_allocation.split(":")
                    node = next(n for n in self.nodes if n.node_id == node_id)
                    node.release_gpus(job.job_id)
        
        # Remove completed jobs from running
        for job_id in completed_jobs:
            del self.running_jobs[job_id]
    
    def get_gpu_utilization(self) -> float:
        """Get current GPU utilization percentage"""
        total_allocated = sum(node.num_gpu - node.free_gpus for node in self.nodes)
        return (total_allocated / self.total_gpus) * 100


class JobGenerator:
    """Generates realistic job workloads for testing"""
    
    def __init__(self):
        self.model_names = [
            "resnet50", "resnet101", "resnet152",
            "bert-base", "bert-large", "roberta-base",
            "transformer-xl", "gpt2", "gpt3",
            "lstm-sentiment", "lstm-translation",
            "vgg16", "inception-v3", "efficientnet-b0"
        ]
    
    def generate_jobs(self, num_jobs: int, time_range: float = 86400) -> List[Job]:
        """Generate a realistic workload of jobs"""
        jobs = []
        
        for i in range(num_jobs):
            # Random job characteristics
            num_gpu = random.choice([1, 2, 4, 8])
            iterations = random.randint(100, 10000)
            model_name = random.choice(self.model_names)
            duration = random.uniform(300, 7200)  # 5 min to 2 hours
            interval = random.uniform(0.1, 1.0)
            
            # Submit time spread over the time range (starting from 0)
            submit_time = random.uniform(0, time_range)
            
            job = Job(
                job_id=f"job_{i:06d}",
                num_gpu=num_gpu,
                submit_time=submit_time,
                iterations=iterations,
                model_name=model_name,
                duration=duration,
                interval=interval
            )
            jobs.append(job)
        
        # Sort by submit time
        jobs.sort(key=lambda j: j.submit_time)
        return jobs


class SchedulerTester:
    """Tests and compares all schedulers"""
    
    def __init__(self, num_jobs: int = 5000):
        self.num_jobs = num_jobs
        self.job_generator = JobGenerator()
        self.schedulers = {
            "FIFO": FIFOScheduler(),
            "SJF": SJFScheduler(),
            "Shortest": ShortestScheduler(),
            "Shortest-GPU": ShortestGPUScheduler(),
            "Hybrid-Priority": HybridPriorityScheduler(),
            "Predictive-Backfill": PredictiveBackfillScheduler(),
            "Smart-Batch": SmartBatchScheduler()
        }
        self.results: Dict[str, SchedulerResult] = {}
    
    def test_scheduler(self, scheduler_name: str, scheduler: Any, jobs: List[Job]) -> SchedulerResult:
        """Test a single scheduler with the given jobs"""
        print(f"Testing {scheduler_name}...")
        
        # Create a fresh cluster for this test
        cluster = ClusterSimulator()
        
        # Copy jobs to avoid modifying originals
        test_jobs = [Job(
            job_id=job.job_id,
            num_gpu=job.num_gpu,
            submit_time=job.submit_time,
            iterations=job.iterations,
            model_name=job.model_name,
            duration=job.duration,
            interval=job.interval
        ) for job in jobs]
        
        pending_jobs = test_jobs.copy()
        completed_jobs = []
        wait_times = []
        execution_times = []
        
        # Simulation loop
        time_step = 1.0  # 1 second steps
        max_simulation_time = 86400 * 7  # 7 days max
        
        while (pending_jobs or cluster.running_jobs) and cluster.current_time < max_simulation_time:
            # Add new jobs that have arrived
            current_jobs = [j for j in pending_jobs if j.submit_time <= cluster.current_time]
            for job in current_jobs:
                if job in pending_jobs:
                    pending_jobs.remove(job)
            
            # Try to schedule jobs
            while current_jobs and cluster.can_allocate_job(current_jobs[0]):
                job = scheduler.select_job(current_jobs)
                if job and job in current_jobs:
                    if cluster.allocate_job(job):
                        current_jobs.remove(job)
                        # Calculate wait time
                        wait_time = cluster.current_time - job.submit_time
                        wait_times.append(wait_time)
                    else:
                        break
                else:
                    break
            
            # Advance time
            cluster.advance_time(time_step)
            
            # Move completed jobs
            for job in cluster.completed_jobs:
                if job not in completed_jobs:
                    completed_jobs.append(job)
                    execution_times.append(job.execution_time)
        
        # Calculate metrics
        total_time = cluster.current_time
        avg_wait_time = statistics.mean(wait_times) if wait_times else 0
        avg_execution_time = statistics.mean(execution_times) if execution_times else 0
        gpu_utilization = cluster.get_gpu_utilization()
        throughput = len(completed_jobs) / (total_time / 3600) if total_time > 0 else 0
        
        # Calculate fairness (lower is better - based on wait time variance)
        fairness_score = statistics.variance(wait_times) if len(wait_times) > 1 else 0
        
        # Job completion times
        job_completion_times = [j.total_time for j in completed_jobs]
        
        return SchedulerResult(
            scheduler_name=scheduler_name,
            total_jobs=len(test_jobs),
            completed_jobs=len(completed_jobs),
            total_time=total_time,
            avg_wait_time=avg_wait_time,
            avg_execution_time=avg_execution_time,
            gpu_utilization=gpu_utilization,
            throughput=throughput,
            fairness_score=fairness_score,
            job_completion_times=job_completion_times,
            wait_times=wait_times,
            execution_times=execution_times
        )
    
    def run_all_tests(self) -> Dict[str, SchedulerResult]:
        """Run tests for all schedulers"""
        print(f"Generating {self.num_jobs} test jobs...")
        jobs = self.job_generator.generate_jobs(self.num_jobs)
        
        print(f"Starting scheduler tests...")
        for scheduler_name, scheduler in self.schedulers.items():
            try:
                result = self.test_scheduler(scheduler_name, scheduler, jobs)
                self.results[scheduler_name] = result
                print(f"✓ {scheduler_name}: {result.completed_jobs}/{result.total_jobs} jobs completed")
            except Exception as e:
                print(f"✗ {scheduler_name} failed: {e}")
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate a comprehensive performance report"""
        if not self.results:
            return "No test results available"
        
        report = []
        report.append("=" * 80)
        report.append("SCHEDULER PERFORMANCE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Total Jobs Tested: {self.num_jobs}")
        report.append(f"Number of Schedulers: {len(self.results)}")
        report.append("")
        
        # Summary table
        report.append("PERFORMANCE SUMMARY")
        report.append("-" * 80)
        report.append(f"{'Scheduler':<20} {'Completed':<10} {'Throughput':<12} {'Avg Wait':<10} {'GPU Util':<10}")
        report.append("-" * 80)
        
        for name, result in self.results.items():
            report.append(f"{name:<20} {result.completed_jobs:<10} {result.throughput:<12.1f} "
                         f"{result.avg_wait_time:<10.1f} {result.gpu_utilization:<10.1f}")
        
        report.append("")
        
        # Detailed analysis
        report.append("DETAILED ANALYSIS")
        report.append("-" * 80)
        
        # Find best performers
        best_throughput = max(self.results.values(), key=lambda r: r.throughput)
        best_wait_time = min(self.results.values(), key=lambda r: r.avg_wait_time)
        best_gpu_util = max(self.results.values(), key=lambda r: r.gpu_utilization)
        best_fairness = min(self.results.values(), key=lambda r: r.fairness_score)
        
        report.append(f"Best Throughput: {best_throughput.scheduler_name} ({best_throughput.throughput:.1f} jobs/hour)")
        report.append(f"Best Wait Time: {best_wait_time.scheduler_name} ({best_wait_time.avg_wait_time:.1f}s)")
        report.append(f"Best GPU Utilization: {best_gpu_util.scheduler_name} ({best_gpu_util.gpu_utilization:.1f}%)")
        report.append(f"Best Fairness: {best_fairness.scheduler_name} (variance: {best_fairness.fairness_score:.1f})")
        
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 80)
        
        if best_throughput.scheduler_name == best_wait_time.scheduler_name:
            report.append(f" {best_throughput.scheduler_name} appears to be the best overall performer")
        else:
            report.append(f" For maximum throughput: {best_throughput.scheduler_name}")
            report.append(f" For minimum wait times: {best_wait_time.scheduler_name}")
        
        report.append(f" For maximum GPU utilization: {best_gpu_util.scheduler_name}")
        report.append(f"  For fairest scheduling: {best_fairness.scheduler_name}")
        
        return "\n".join(report)
    
    def create_visualizations(self):
        """Create performance comparison charts"""
        if not self.results:
            print("No results to visualize")
            return
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Scheduler Performance Comparison ({self.num_jobs} jobs)', fontsize=16)
        
        # Extract data
        names = list(self.results.keys())
        throughputs = [r.throughput for r in self.results.values()]
        wait_times = [r.avg_wait_time for r in self.results.values()]
        gpu_utils = [r.gpu_utilization for r in self.results.values()]
        fairness_scores = [r.fairness_score for r in self.results.values()]
        
        # Throughput comparison
        bars1 = ax1.bar(names, throughputs, color='skyblue', alpha=0.7)
        ax1.set_title('Throughput (jobs/hour)')
        ax1.set_ylabel('Jobs per Hour')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, throughputs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # Wait time comparison
        bars2 = ax2.bar(names, wait_times, color='lightcoral', alpha=0.7)
        ax2.set_title('Average Wait Time')
        ax2.set_ylabel('Seconds')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars2, wait_times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # GPU utilization comparison
        bars3 = ax3.bar(names, gpu_utils, color='lightgreen', alpha=0.7)
        ax3.set_title('GPU Utilization')
        ax3.set_ylabel('Percentage (%)')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars3, gpu_utils):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # Fairness comparison (lower is better)
        bars4 = ax4.bar(names, fairness_scores, color='gold', alpha=0.7)
        ax4.set_title('Fairness Score (Variance - Lower is Better)')
        ax4.set_ylabel('Variance')
        ax4.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars4, fairness_scores):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('scheduler_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(" Visualization saved as 'scheduler_performance_comparison.png'")


def main():
    """Main testing function"""
    print(" Starting Comprehensive Scheduler Performance Testing")
    print("=" * 60)
    
    # Create tester
    tester = SchedulerTester(num_jobs=100)
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Generate report
    report = tester.generate_report()
    print("\n" + report)
    
    # Save report to file
    with open('scheduler_performance_report.txt', 'w') as f:
        f.write(report)
    print("\n Report saved to 'scheduler_performance_report.txt'")
    
    # Create visualizations
    tester.create_visualizations()
    
    # Save detailed results to CSV
    detailed_results = []
    for name, result in results.items():
        detailed_results.append({
            'Scheduler': name,
            'Total_Jobs': result.total_jobs,
            'Completed_Jobs': result.completed_jobs,
            'Total_Time': result.total_time,
            'Avg_Wait_Time': result.avg_wait_time,
            'Avg_Execution_Time': result.avg_execution_time,
            'GPU_Utilization': result.gpu_utilization,
            'Throughput': result.throughput,
            'Fairness_Score': result.fairness_score
        })
    
    df = pd.DataFrame(detailed_results)
    df.to_csv('scheduler_detailed_results.csv', index=False)
    print("Detailed results saved to 'scheduler_detailed_results.csv'")
    
    print("\nTesting completed successfully!")


if __name__ == "__main__":
    main() 