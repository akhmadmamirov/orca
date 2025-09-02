#!/usr/bin/env python3
"""
Test script to demonstrate actual scheduler differences
Creates resource contention to force schedulers to make meaningful decisions
"""

from core import GPUClusterManager
import time


def test_resource_contention():
    """Test schedulers with limited resources to show real differences"""
    print("Testing Scheduler Differences with Resource Contention")
    print("=" * 70)
    
    schedulers = ['fifo', 'sjf', 'shortest', 'shortest-gpu']
    results = {}
    
    for scheduler in schedulers:
        print(f"\n{'='*25} Testing {scheduler.upper()} Scheduler {'='*25}")
        
        # Create a SMALLER cluster to force resource contention
        # 2 nodes × 2 GPUs = 4 total GPUs available
        cluster = GPUClusterManager(num_nodes=2, gpus_per_node=2)
        cluster.set_scheduler(scheduler)
        
        print(f"Cluster: 2 nodes × 2 GPUs = 4 total GPUs")
        
        # Submit jobs that CANNOT all run simultaneously
        # Total needed: 2+2+1+1 = 6 GPUs, but only 4 available
        print(f"Submitting jobs requiring 6 GPUs total (resource contention!)")
        
        cluster.submit_job(num_gpu=2, iterations=1000, model_name="LargeJob", duration=25, submit_time=0.0)
        cluster.submit_job(num_gpu=2, iterations=800, model_name="MediumJob", duration=20, submit_time=2.0)
        cluster.submit_job(num_gpu=1, iterations=500, model_name="SmallJob", duration=10, submit_time=5.0)
        cluster.submit_job(num_gpu=1, iterations=300, model_name="TinyJob", duration=8, submit_time=8.0)
        
        print(f"Jobs submitted:")
        print(f"  LargeJob: 2 GPUs, 25s duration, submit_time=0.0s")
        print(f"  MediumJob: 2 GPUs, 20s duration, submit_time=2.0s")
        print(f"  SmallJob: 1 GPU, 10s duration, submit_time=5.0s")
        print(f"  TinyJob: 1 GPU, 8s duration, submit_time=8.0s")
        
        # Run simulation until all jobs complete
        print(f"\nRunning simulation...")
        for step in range(60):
            cluster.update_simulation(1.0)
            if len(cluster.completed_jobs) == 4:
                break
        
        # Record detailed results
        avg_jct = cluster.metrics.get_average_jct()
        gpu_util = cluster.metrics.get_gpu_utilization()
        fragmentation = cluster.metrics.get_resource_fragmentation()
        
        results[scheduler] = {
            'jct': avg_jct,
            'utilization': gpu_util,
            'fragmentation': fragmentation,
            'completion_order': [job.job_id for job in cluster.completed_jobs],
            'total_time': cluster.simulation_time
        }
        
        print(f"\nResults with {scheduler}:")
        print(f"  Average JCT: {avg_jct:.2f}s")
        print(f"  GPU Utilization: {gpu_util:.1f}%")
        print(f"  Resource Fragmentation: {fragmentation:.2f}")
        print(f"  Total Simulation Time: {cluster.simulation_time:.1f}s")
        print(f"  Job Completion Order: {[job.job_id for job in cluster.completed_jobs]}")
        
        # Show detailed timing for each job
        print(f"  Job Details:")
        for job in cluster.completed_jobs:
            jct = cluster.metrics.get_job_completion_time(job.job_id)
            wait_time = jct - job.duration
            print(f"    {job.job_id}: {job.model_name}, {job.num_gpu} GPUs, {job.duration}s duration")
            print(f"         → JCT: {jct:.1f}s, Wait Time: {wait_time:.1f}s")
        
        print()
    
    # Compare results
    print(f"\n{'='*70}")
    print(" zSCHEDULER COMPARISON RESULTS")
    print("=" * 70)
    
    print(f"\n Performance Rankings:")
    
    # JCT ranking (lower is better)
    jct_ranking = sorted(results.items(), key=lambda x: x[1]['jct'])
    print(f"  Best Average JCT: {jct_ranking[0][0].upper()} ({jct_ranking[0][1]['jct']:.2f}s)")
    print(f"  Worst Average JCT: {jct_ranking[-1][0].upper()} ({jct_ranking[-1][1]['jct']:.2f}s)")
    
    # Total time ranking (lower is better)
    time_ranking = sorted(results.items(), key=lambda x: x[1]['total_time'])
    print(f"  Fastest Total Completion: {time_ranking[0][0].upper()} ({time_ranking[0][1]['total_time']:.1f}s)")
    print(f"  Slowest Total Completion: {time_ranking[-1][0].upper()} ({time_ranking[-1][1]['total_time']:.1f}s)")
    
    # Utilization ranking (higher is better)
    util_ranking = sorted(results.items(), key=lambda x: x[1]['utilization'], reverse=True)
    print(f"  Best GPU Utilization: {util_ranking[0][0].upper()} ({util_ranking[0][1]['utilization']:.1f}%)")
    print(f"  Worst GPU Utilization: {util_ranking[-1][0].upper()} ({util_ranking[-1][1]['utilization']:.1f}%)")
    
    print(f"\n📋 Detailed Comparison:")
    for scheduler, result in results.items():
        print(f"  {scheduler.upper()}:")
        print(f"    Avg JCT: {result['jct']:.2f}s (Rank: {jct_ranking.index((scheduler, result)) + 1})")
        print(f"    Total Time: {result['total_time']:.1f}s (Rank: {time_ranking.index((scheduler, result)) + 1})")
        print(f"    GPU Util: {result['utilization']:.1f}% (Rank: {util_ranking.index((scheduler, result)) + 1})")
        print(f"    Completion Order: {result['completion_order']}")
        print()
    
    # Explain why schedulers differ
    print(f" Why Schedulers Differ in This Test:")
    print(f"  1. Resource Contention: 6 GPUs needed, only 4 available")
    print(f"  2. Job Queuing: Some jobs must wait for resources")
    print(f"  3. Scheduler Decisions: Must choose which jobs to run first")
    print(f"  4. Resource Efficiency: Better scheduling = faster completion")
    
    return results


def test_fifo_vs_sjf_scenario():
    """Specific test to show FIFO vs SJF differences"""
    print(f"\n{'='*70}")
    print("⚡ FIFO vs SJF Specific Comparison")
    print("=" * 70)
    
    # Test scenario: 2 large jobs vs 1 small job
    # FIFO should run in order, SJF should prioritize small job
    
    schedulers = ['fifo', 'sjf']
    
    for scheduler in schedulers:
        print(f"\n{'='*20} {scheduler.upper()} Scheduler {'='*20}")
        
        cluster = GPUClusterManager(num_nodes=1, gpus_per_node=2)  # Only 2 GPUs total
        cluster.set_scheduler(scheduler)
        
        print(f"Cluster: 1 node × 2 GPUs = 2 total GPUs")
        
        # Submit 3 jobs that can't all run simultaneously
        cluster.submit_job(num_gpu=2, iterations=1000, model_name="BigJob", duration=30, submit_time=0.0)
        cluster.submit_job(num_gpu=1, iterations=200, model_name="SmallJob", duration=5, submit_time=2.0)
        cluster.submit_job(num_gpu=2, iterations=800, model_name="AnotherBigJob", duration=25, submit_time=5.0)
        
        print(f"Jobs:")
        print(f"  BigJob: 2 GPUs, 30s duration, submit_time=0.0s")
        print(f"  SmallJob: 1 GPU, 5s duration, submit_time=2.0s")
        print(f"  AnotherBigJob: 2 GPUs, 25s duration, submit_time=5.0s")
        
        # Run simulation
        for step in range(80):
            cluster.update_simulation(1.0)
            if len(cluster.completed_jobs) == 3:
                break
        
        # Show results
        print(f"\nResults:")
        print(f"  Average JCT: {cluster.metrics.get_average_jct():.2f}s")
        print(f"  Total Time: {cluster.simulation_time:.1f}s")
        print(f"  Completion Order: {[job.job_id for job in cluster.completed_jobs]}")
        
        # Show what happened to each job
        for job in cluster.completed_jobs:
            jct = cluster.metrics.get_job_completion_time(job.job_id)
            wait_time = jct - job.duration
            print(f"    {job.job_id}: {job.model_name} - JCT: {jct:.1f}s, Wait: {wait_time:.1f}s")
    
    print(f"\n Expected Behavior:")
    print(f"  FIFO: Should run BigJob first (submit_time=0.0), then SmallJob, then AnotherBigJob")
    print(f"  SJF: Should prioritize SmallJob (1 GPU) over the 2-GPU jobs")
    print(f"  Result: Different completion orders and JCTs")


if __name__ == "__main__":
    print("Testing Real Scheduler Differences with Resource Contention")
    print("=" * 70)
    
    # Test with resource contention
    results = test_resource_contention()
    
    # Test specific FIFO vs SJF scenario
    test_fifo_vs_sjf_scenario()
    
    print(f"\n Scheduler difference testing completed!")
    print("=" * 70) 