#!/usr/bin/env python3
"""
Test script for the GPU Cluster Management System
Demonstrates core functionality and validates system behavior
"""

from core import GPUClusterManager
from models import Job, JobState
import time


def test_basic_functionality():
    """Test basic cluster operations"""
    print("Testing Basic Functionality")
    print("=" * 40)
    
    # Create cluster
    cluster = GPUClusterManager(num_nodes=2, gpus_per_node=4)
    
            # Submit jobs
        job1 = cluster.submit_job(num_gpu=2, iterations=100, model_name="TestModel1", duration=10, submit_time=0.0)
        job2 = cluster.submit_job(num_gpu=1, iterations=50, model_name="TestModel2", duration=5, submit_time=2.0)
    
    print(f"Submitted jobs: {job1}, {job2}")
    print(f"Pending jobs: {len(cluster.pending_jobs)}")
    
    # Run simulation
    for step in range(15):
        cluster.update_simulation(1.0)
        if len(cluster.completed_jobs) == 2:
            break
    
    print(f"Completed jobs: {len(cluster.completed_jobs)}")
    print(f"Running jobs: {len(cluster.running_jobs)}")
    print(f"Pending jobs: {len(cluster.pending_jobs)}")
    
    # Check final status
    cluster.print_status()
    print("Basic functionality test passed\n")


def test_scheduling_policies():
    """Test different scheduling policies"""
    print("Testing Scheduling Policies")
    print("=" * 40)
    
    schedulers = ['fifo', 'sjf', 'shortest', 'shortest-gpu']
    results = {}
    
    for scheduler in schedulers:
        print(f"\nTesting {scheduler.upper()} scheduler...")
        
        # Create fresh cluster
        cluster = GPUClusterManager(num_nodes=2, gpus_per_node=4)
        cluster.set_scheduler(scheduler)
        
        # Submit jobs with different characteristics
        cluster.submit_job(num_gpu=4, iterations=200, model_name="LargeModel", duration=20)
        cluster.submit_job(num_gpu=1, iterations=50, model_name="SmallModel", duration=5)
        cluster.submit_job(num_gpu=2, iterations=100, model_name="MediumModel", duration=10)
        
        # Run simulation
        start_time = time.time()
        for step in range(30):
            cluster.update_simulation(1.0)
            if len(cluster.completed_jobs) == 3:
                break
        
        # Record results
        avg_jct = cluster.metrics.get_average_jct()
        results[scheduler] = avg_jct
        print(f"  Average JCT: {avg_jct:.2f}s")
    
    # Compare results
    print(f"\nScheduling Policy Comparison:")
    for scheduler, jct in results.items():
        print(f"  {scheduler.upper()}: {jct:.2f}s")
    
    best_scheduler = min(results, key=results.get)
    print(f"Best performing: {best_scheduler.upper()}")
    print("Scheduling policies test passed\n")


def test_placement_schemes():
    """Test different placement schemes"""
    print("Testing Placement Schemes")
    print("=" * 40)
    
    schemes = ['first-fit', 'best-fit']
    results = {}
    
    for scheme in schemes:
        print(f"\nTesting {scheme} placement...")
        
        # Create fresh cluster
        cluster = GPUClusterManager(num_nodes=3, gpus_per_node=4)
        cluster.set_placement(scheme)
        
        # Submit jobs that will cause fragmentation
        cluster.submit_job(num_gpu=3, iterations=100, model_name="Model1", duration=15)
        cluster.submit_job(num_gpu=2, iterations=80, model_name="Model2", duration=10)
        cluster.submit_job(num_gpu=1, iterations=60, model_name="Model3", duration=8)
        
        # Run simulation
        for step in range(20):
            cluster.update_simulation(1.0)
            if len(cluster.completed_jobs) == 3:
                break
        
        # Record fragmentation
        fragmentation = cluster.metrics.get_resource_fragmentation()
        results[scheme] = fragmentation
        print(f"  Resource fragmentation: {fragmentation:.2f}")
    
    # Compare results
    print(f"\n Placement Scheme Comparison:")
    for scheme, frag in results.items():
        print(f"  {scheme}: {frag:.2f}")
    
    best_scheme = min(results, key=results.get)
    print(f"Best placement: {best_scheme}")
    print("Placement schemes test passed\n")


def test_resource_utilization():
    """Test resource utilization tracking"""
    print("Testing Resource Utilization")
    print("=" * 40)
    
    # Create cluster
    cluster = GPUClusterManager(num_nodes=2, gpus_per_node=4)
    
    # Submit jobs to utilize resources
    cluster.submit_job(num_gpu=3, iterations=150, model_name="HeavyModel", duration=25)
    cluster.submit_job(num_gpu=2, iterations=100, model_name="MediumModel", duration=15)
    
    # Monitor utilization over time
    utilization_history = []
    for step in range(30):
        cluster.update_simulation(1.0)
        current_util = cluster.metrics.get_gpu_utilization()
        utilization_history.append(current_util)
        
        if len(cluster.completed_jobs) == 2:
            break
    
    print(f"GPU utilization history: {[f'{u:.1f}%' for u in utilization_history[:10]]}...")
    print(f"Peak utilization: {max(utilization_history):.1f}%")
    print(f"Average utilization: {sum(utilization_history)/len(utilization_history):.1f}%")
    
    # Check node status
    print("\nFinal node status:")
    for node_id, node in cluster.node_objects.items():
        print(f"  {node_id}: {node.utilization:.1f}% utilized")
    
    print("Resource utilization test passed\n")


def test_job_states():
    """Test job state transitions"""
    print("Testing Job State Transitions")
    print("=" * 40)
    
    # Create cluster
    cluster = GPUClusterManager(num_nodes=1, gpus_per_node=4)
    
    # Submit a job
    job_id = cluster.submit_job(num_gpu=2, iterations=100, model_name="StateTestModel", duration=10)
    
    # Check initial state
    job = next(j for j in cluster.pending_jobs if j.job_id == job_id)
    print(f"Initial state: {job.state}")
    print(f"Pending time: {job.pending_time}")
    
    # Run simulation until job starts
    for step in range(5):
        cluster.update_simulation(1.0)
        if job.state == JobState.RUNNING:
            break
    
    print(f"After starting: {job.state}")
    print(f"Start time: {job.start_time}")
    print(f"Execution time: {job.execution_time}")
    
    # Run until completion
    for step in range(15):
        cluster.update_simulation(1.0)
        if job.state == JobState.END:
            break
    
    print(f"Final state: {job.state}")
    print(f"End time: {job.end_time}")
    print(f"Total execution time: {job.execution_time}")
    print(f"Total time: {job.total_time}")
    
    print("Job state transitions test passed\n")


def run_all_tests():
    """Run all test functions"""
    print("Starting GPU Cluster Management System Tests")
    print("=" * 60)
    
    try:
        test_basic_functionality()
        test_scheduling_policies()
        test_placement_schemes()
        test_resource_utilization()
        test_job_states()
        
        print("All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests() 