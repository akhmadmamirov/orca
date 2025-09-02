#!/usr/bin/env python3
"""
Example usage of the GPU Cluster Management System
Demonstrates various use cases and configurations
"""

from core import GPUClusterManager
import time


def example_basic_workflow():
    """Basic workflow example"""
    print("🚀 Basic Workflow Example")
    print("=" * 40)
    
    # Create a small cluster
    cluster = GPUClusterManager(num_nodes=2, gpus_per_node=4)
    
    # Submit some deep learning jobs
    cluster.submit_job(num_gpu=2, iterations=1000, model_name="ResNet50", duration=20, submit_time=0.0)
    cluster.submit_job(num_gpu=1, iterations=500, model_name="VGG16", duration=10, submit_time=3.0)
    cluster.submit_job(num_gpu=3, iterations=1500, model_name="BERT", duration=30, submit_time=8.0)
    
    print(f"Submitted {len(cluster.pending_jobs)} jobs")
    
    # Run simulation
    for step in range(35):
        cluster.update_simulation(1.0)
        if len(cluster.completed_jobs) == 3:
            break
    
    # Check results
    cluster.print_status()
    print()


def example_scheduler_comparison():
    """Compare different scheduling policies"""
    print("📊 Scheduler Comparison Example")
    print("=" * 40)
    
    schedulers = ['fifo', 'sjf', 'shortest', 'shortest-gpu']
    results = {}
    
    for scheduler in schedulers:
        print(f"\nTesting {scheduler.upper()}...")
        
        # Create fresh cluster
        cluster = GPUClusterManager(num_nodes=3, gpus_per_node=4)
        cluster.set_scheduler(scheduler)
        
        # Submit mixed workload
        cluster.submit_job(num_gpu=4, iterations=2000, model_name="LargeModel", duration=40, submit_time=0.0)
        cluster.submit_job(num_gpu=1, iterations=300, model_name="SmallModel", duration=8, submit_time=2.0)
        cluster.submit_job(num_gpu=2, iterations=800, model_name="MediumModel", duration=15, submit_time=5.0)
        cluster.submit_job(num_gpu=3, iterations=1200, model_name="Transformer", duration=25, submit_time=10.0)
        
        # Run simulation
        for step in range(50):
            cluster.update_simulation(1.0)
            if len(cluster.completed_jobs) == 4:
                break
        
        # Record metrics
        avg_jct = cluster.metrics.get_average_jct()
        gpu_util = cluster.metrics.get_gpu_utilization()
        fragmentation = cluster.metrics.get_resource_fragmentation()
        
        results[scheduler] = {
            'jct': avg_jct,
            'utilization': gpu_util,
            'fragmentation': fragmentation
        }
        
        print(f"  JCT: {avg_jct:.2f}s, GPU Util: {gpu_util:.1f}%, Frag: {fragmentation:.2f}")
    
    # Find best scheduler for each metric
    print(f"\n🏆 Best Performers:")
    best_jct = min(results, key=lambda x: results[x]['jct'])
    best_util = max(results, key=lambda x: results[x]['utilization'])
    best_frag = min(results, key=lambda x: results[x]['fragmentation'])
    
    print(f"  Best JCT: {best_jct.upper()}")
    print(f"  Best GPU Utilization: {best_util.upper()}")
    print(f"  Best Resource Fragmentation: {best_frag.upper()}")
    print()


def example_placement_optimization():
    """Demonstrate placement scheme optimization"""
    print("🎯 Placement Optimization Example")
    print("=" * 40)
    
    schemes = ['first-fit', 'best-fit']
    
    for scheme in schemes:
        print(f"\nTesting {scheme} placement...")
        
        # Create cluster with specific configuration
        cluster = GPUClusterManager(num_nodes=4, gpus_per_node=4)
        cluster.set_placement(scheme)
        
        # Submit jobs that will test placement efficiency
        cluster.submit_job(num_gpu=3, iterations=1000, model_name="ModelA", duration=20)
        cluster.submit_job(num_gpu=2, iterations=800, model_name="ModelB", duration=15)
        cluster.submit_job(num_gpu=1, iterations=400, model_name="ModelC", duration=8)
        cluster.submit_job(num_gpu=4, iterations=1500, model_name="ModelD", duration=30)
        cluster.submit_job(num_gpu=2, iterations=600, model_name="ModelE", duration=12)
        
        # Run simulation
        for step in range(40):
            cluster.update_simulation(1.0)
            if len(cluster.completed_jobs) == 5:
                break
        
        # Show placement results
        print(f"  Jobs completed: {len(cluster.completed_jobs)}")
        print(f"  Final GPU utilization: {cluster.metrics.get_gpu_utilization():.1f}%")
        print(f"  Resource fragmentation: {cluster.metrics.get_resource_fragmentation():.2f}")
        
        # Show node utilization
        print("  Node utilization:")
        for node_id, node in cluster.node_objects.items():
            print(f"    {node_id}: {node.utilization:.1f}%")
    
    print()


def example_dynamic_workload():
    """Demonstrate handling dynamic workload"""
    print("⚡ Dynamic Workload Example")
    print("=" * 40)
    
    cluster = GPUClusterManager(num_nodes=2, gpus_per_node=4)
    
    # Initial jobs
    cluster.submit_job(num_gpu=2, iterations=800, model_name="InitialJob1", duration=15)
    cluster.submit_job(num_gpu=1, iterations=400, model_name="InitialJob2", duration=8)
    
    print("Initial workload submitted")
    cluster.print_status()
    
    # Run for a while
    for step in range(10):
        cluster.update_simulation(1.0)
    
    print(f"\nAfter 10 time steps:")
    cluster.print_status()
    
    # Add more jobs dynamically
    print(f"\nAdding dynamic workload...")
    cluster.submit_job(num_gpu=3, iterations=1200, model_name="DynamicJob1", duration=25)
    cluster.submit_job(num_gpu=1, iterations=300, model_name="DynamicJob2", duration=6)
    
    # Continue simulation
    for step in range(30):
        cluster.update_simulation(1.0)
        if len(cluster.completed_jobs) == 4:
            break
    
    print(f"\nFinal status:")
    cluster.print_status()
    print()


def example_custom_analysis():
    """Custom analysis example"""
    print("🔍 Custom Analysis Example")
    print("=" * 40)
    
    cluster = GPUClusterManager(num_nodes=3, gpus_per_node=4)
    
    # Submit diverse workload
    job_ids = []
    job_ids.append(cluster.submit_job(num_gpu=1, iterations=200, model_name="TinyModel", duration=5))
    job_ids.append(cluster.submit_job(num_gpu=2, iterations=600, model_name="SmallModel", duration=12))
    job_ids.append(cluster.submit_job(num_gpu=3, iterations=1000, model_name="MediumModel", duration=20))
    job_ids.append(cluster.submit_job(num_gpu=4, iterations=1500, model_name="LargeModel", duration=35))
    
    # Run simulation
    for step in range(40):
        cluster.update_simulation(1.0)
        if len(cluster.completed_jobs) == 4:
            break
    
    # Custom analysis
    print("Job completion analysis:")
    for job_id in job_ids:
        jct = cluster.metrics.get_job_completion_time(job_id)
        if jct:
            print(f"  {job_id}: {jct:.2f}s")
    
    # Resource efficiency analysis
    total_gpus = sum(node.num_gpu for node in cluster.node_objects.values())
    peak_utilization = max(cluster.metrics.cluster_metrics.get('gpu_utilization', [0]))
    
    print(f"\nResource efficiency:")
    print(f"  Total GPUs: {total_gpus}")
    print(f"  Peak utilization: {peak_utilization:.1f}%")
    print(f"  Average JCT: {cluster.metrics.get_average_jct():.2f}s")
    print(f"  Resource fragmentation: {cluster.metrics.get_resource_fragmentation():.2f}")
    
    # Calculate throughput
    total_time = cluster.simulation_time
    throughput = len(cluster.completed_jobs) / total_time if total_time > 0 else 0
    print(f"  Job throughput: {throughput:.3f} jobs/second")
    print()


def main():
    """Run all examples"""
    print("🎯 GPU Cluster Management System - Examples")
    print("=" * 60)
    
    examples = [
        example_basic_workflow,
        example_scheduler_comparison,
        example_placement_optimization,
        example_dynamic_workload,
        example_custom_analysis
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n📝 Example {i}")
        print("-" * 30)
        try:
            example()
        except Exception as e:
            print(f"❌ Example {i} failed: {e}")
        
        if i < len(examples):
            input("Press Enter to continue to next example...")
    
    print("\n🎉 All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main() 