#!/usr/bin/env python3
"""
Demonstration of the new Adaptive Multi-Factor Scheduler (AMFS)
Shows how it intelligently balances multiple objectives and adapts to system conditions
"""

import time
import random
from models.job import Job, JobState
from schedulers import (
    FIFOScheduler, 
    SJFScheduler, 
    ShortestScheduler, 
    ShortestGPUScheduler,
    AdaptiveMultiFactorScheduler
)


def create_sample_jobs():
    """Create a diverse set of sample jobs for testing"""
    jobs = []
    current_time = time.time()
    
    # Job 1: Small, short job
    jobs.append(Job(
        job_id="job_001",
        num_gpu=2,
        submit_time=current_time - 60,  # Submitted 1 minute ago
        iterations=100,
        model_name="resnet18",
        duration=300,  # 5 minutes
        interval=1.0
    ))
    
    # Job 2: Medium, medium job
    jobs.append(Job(
        job_id="job_002", 
        num_gpu=4,
        submit_time=current_time - 120,  # Submitted 2 minutes ago
        iterations=500,
        model_name="resnet50",
        duration=900,  # 15 minutes
        interval=1.0
    ))
    
    # Job 3: Large, long job
    jobs.append(Job(
        job_id="job_003",
        num_gpu=8,
        submit_time=current_time - 180,  # Submitted 3 minutes ago
        iterations=1000,
        model_name="bert-large",
        duration=1800,  # 30 minutes
        interval=1.0
    ))
    
    # Job 4: Small, long job
    jobs.append(Job(
        job_id="job_004",
        num_gpu=1,
        submit_time=current_time - 240,  # Submitted 4 minutes ago
        iterations=2000,
        model_name="lstm",
        duration=1200,  # 20 minutes
        interval=1.0
    ))
    
    # Job 5: Large, short job
    jobs.append(Job(
        job_id="job_005",
        num_gpu=6,
        submit_time=current_time - 300,  # Submitted 5 minutes ago
        iterations=300,
        model_name="transformer",
        duration=600,  # 10 minutes
        interval=1.0
    ))
    
    return jobs


def simulate_scheduling(scheduler, jobs, scenario_name):
    """Simulate scheduling decisions for a given scenario"""
    print(f"\n=== {scenario_name} ===")
    print(f"Scheduler: {scheduler.name}")
    
    # Simulate different queue lengths
    queue_scenarios = [
        ("Short Queue (3 jobs)", jobs[:3]),
        ("Medium Queue (4 jobs)", jobs[:4]), 
        ("Long Queue (5 jobs)", jobs)
    ]
    
    for queue_name, queue_jobs in queue_scenarios:
        print(f"\n{queue_name}:")
        selected_job = scheduler.select_job(queue_jobs)
        if selected_job:
            print(f"  Selected: {selected_job.job_id} (GPUs: {selected_job.num_gpu}, Duration: {selected_job.duration}s, Wait: {time.time() - selected_job.submit_time:.0f}s)")
            
            # Show scheduler info if available
            if hasattr(scheduler, 'get_scheduler_info'):
                info = scheduler.get_scheduler_info()
                print(f"  Weights: Efficiency={info['efficiency_weight']:.2f}, Fairness={info['fairness_weight']:.2f}, Resource={info['resource_weight']:.2f}")
                print(f"  Aging Factor: {info['aging_factor']:.2f}")
        else:
            print("  No job selected")


def demonstrate_adaptive_behavior():
    """Show how the adaptive scheduler changes behavior based on conditions"""
    print("\n" + "="*60)
    print("ADAPTIVE BEHAVIOR DEMONSTRATION")
    print("="*60)
    
    scheduler = AdaptiveMultiFactorScheduler()
    jobs = create_sample_jobs()
    
    print("\nInitial weights:")
    info = scheduler.get_scheduler_info()
    print(f"Efficiency: {info['efficiency_weight']:.2f}")
    print(f"Fairness: {info['fairness_weight']:.2f}")
    print(f"Resource: {info['resource_weight']:.2f}")
    
    # Show how weights adapt
    print("\nWeight adaptation based on queue length:")
    for i in range(1, 6):
        test_jobs = jobs[:i]
        scheduler.select_job(test_jobs)  # This triggers weight adaptation
        info = scheduler.get_scheduler_info()
        print(f"Queue length {i}: Efficiency={info['efficiency_weight']:.2f}, Fairness={info['fairness_weight']:.2f}, Resource={info['resource_weight']:.2f}")


def compare_schedulers():
    """Compare all schedulers side by side"""
    print("\n" + "="*60)
    print("SCHEDULER COMPARISON")
    print("="*60)
    
    schedulers = [
        FIFOScheduler(),
        SJFScheduler(),
        ShortestScheduler(),
        ShortestGPUScheduler(),
        AdaptiveMultiFactorScheduler()
    ]
    
    jobs = create_sample_jobs()
    
    for scheduler in schedulers:
        simulate_scheduling(scheduler, jobs, f"Testing {scheduler.name}")


def main():
    """Main demonstration function"""
    print("Adaptive Multi-Factor Scheduler (AMFS) Demonstration")
    print("="*60)
    
    print("\nThis new scheduler is superior to existing ones because it:")
    print("Balances multiple objectives (efficiency, fairness, resource utilization)")
    print("Adapts weights based on system conditions (queue length)")
    print("Implements aging to prevent job starvation")
    print("Optimizes GPU utilization while maintaining fairness")
    print("Provides monitoring capabilities for system administrators")
    
    # Compare all schedulers
    compare_schedulers()
    
    # Demonstrate adaptive behavior
    demonstrate_adaptive_behavior()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("The Adaptive Multi-Factor Scheduler provides:")
    print("• Intelligent job selection based on multiple factors")
    print("• Dynamic adaptation to system load")
    print("• Prevention of job starvation through aging")
    print("• Better resource utilization than simple heuristics")
    print("• Configurable parameters for different environments")
    print("• Monitoring capabilities for performance analysis")


if __name__ == "__main__":
    main() 