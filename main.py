#!/usr/bin/env python3
"""
GPU Cluster Management System for Distributed Deep Learning
Main entry point with demo functionality
"""

from core import GPUClusterManager


def demo_simulation():
    """Demonstrate the GPU cluster management system"""
    print("GPU Cluster Management System for Distributed Deep Learning")
    print("="*60)
    
    # Create cluster
    cluster = GPUClusterManager(num_nodes=4, gpus_per_node=4)
    
    # Submit some sample jobs
    print("\nSubmitting sample jobs...")
    cluster.submit_job(num_gpu=2, iterations=1000, model_name="ResNet50", duration=30, submit_time=0.0)
    cluster.submit_job(num_gpu=4, iterations=2000, model_name="BERT", duration=45, submit_time=2.0)
    cluster.submit_job(num_gpu=1, iterations=500, model_name="VGG16", duration=20, submit_time=4.0)
    cluster.submit_job(num_gpu=3, iterations=1500, model_name="Transformer", duration=35, submit_time=6.0)
    cluster.submit_job(num_gpu=2, iterations=800, model_name="EfficientNet", duration=25, submit_time=8.0)
    
    # Run simulation with different schedulers
    schedulers_to_test = ['fifo', 'sjf', 'shortest', 'shortest-gpu']
    
    for scheduler in schedulers_to_test:
        print(f"\n{'='*20} Testing {scheduler.upper()} Scheduler {'='*20}")
        cluster.set_scheduler(scheduler)
        
        # Reset simulation
        cluster.simulation_time = 0
        cluster.pending_jobs = []
        cluster.running_jobs = []
        cluster.completed_jobs = []
        
        # Submit jobs at DIFFERENT times to show scheduler differences
        print("  Submitting jobs at different times...")
        cluster.submit_job(num_gpu=2, iterations=1000, model_name="ResNet50", duration=30, submit_time=0.0)
        cluster.submit_job(num_gpu=4, iterations=2000, model_name="BERT", duration=45, submit_time=5.0)
        cluster.submit_job(num_gpu=1, iterations=500, model_name="VGG16", duration=20, submit_time=8.0)
        cluster.submit_job(num_gpu=3, iterations=1500, model_name="Transformer", duration=35, submit_time=12.0)
        cluster.submit_job(num_gpu=2, iterations=800, model_name="EfficientNet", duration=25, submit_time=15.0)
        
        # Run simulation
        for step in range(50):
            cluster.update_simulation(1.0)
            if len(cluster.completed_jobs) == 5:  # All jobs completed
                break
        
        # Print results
        cluster.print_status()
        print(f"Average JCT with {scheduler}: {cluster.metrics.get_average_jct():.2f}s")


if __name__ == "__main__":
    demo_simulation()
