# GPU Cluster Management System for Distributed Deep Learning

A comprehensive Python implementation for managing GPU clusters, implementing various scheduling policies, placement schemes, and evaluation metrics for distributed deep learning workloads.

## 🎯 Project Overview

This system addresses the challenges of managing GPU clusters for distributed deep learning by providing:

- **System Representation**: Clear modeling of GPUs, nodes, and jobs
- **Scheduling Policies**: Multiple scheduling algorithms to optimize job completion
- **Placement Schemes**: Intelligent GPU allocation strategies
- **Evaluation Metrics**: Comprehensive performance measurement and analysis

## 🏗️ System Architecture

The system follows the workflow illustrated in the block diagram:

```
User → Job Queue → Scheduler → Placement Scheme → GPU Cluster
```

### Core Components

1. **Job Management**: Handles job submission, execution, and completion
2. **Node Management**: Manages GPU allocation and resource tracking
3. **Scheduling Engine**: Implements various job scheduling policies
4. **Placement Engine**: Maps jobs to available GPUs
5. **Metrics Collection**: Tracks performance and utilization metrics

## 🚀 Features

### Job Representation
- **Job States**: ADDED, EVENT, PENDING, RUNNING, END, ERROR
- **Job Properties**: GPU requirements, model information, execution time, iterations
- **Runtime Tracking**: Execution time, pending time, preemption counts

### Scheduling Policies
- **FIFO**: First-in-first-out scheduling
- **SJF**: Smallest job first (by GPU count)
- **Shortest**: Shortest remaining time first
- **Shortest-GPU**: Shortest remaining GPU time first

### Placement Schemes
- **First-Fit**: Allocate to first available node
- **Best-Fit**: Minimize resource fragmentation

### Metrics & Monitoring
- **Job-level**: Completion time, execution time, pending time
- **Cluster-level**: GPU utilization, resource fragmentation, job counts
- **Resource-specific**: CPU, memory, network usage per node

## 📦 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Orca
```

2. Install dependencies (optional packages for enhanced functionality):
```bash
pip install -r requirements.txt
```

## 🎮 Usage

### Basic Usage

```python
from main import GPUClusterManager

# Create a cluster with 4 nodes, 4 GPUs each
cluster = GPUClusterManager(num_nodes=4, gpus_per_node=4)

# Submit jobs
cluster.submit_job(num_gpu=2, iterations=1000, model_name="ResNet50", duration=30)
cluster.submit_job(num_gpu=4, iterations=2000, model_name="BERT", duration=45)

# Run simulation
for step in range(50):
    cluster.update_simulation(1.0)

# Check status
cluster.print_status()
```

### Changing Schedulers

```python
# Switch between different scheduling policies
cluster.set_scheduler('sjf')        # Smallest job first
cluster.set_scheduler('shortest')   # Shortest remaining time
cluster.set_scheduler('fifo')       # First-in-first-out
```

### Changing Placement Schemes

```python
# Switch between placement strategies
cluster.set_placement('first-fit')  # First available node
cluster.set_placement('best-fit')   # Minimize fragmentation
```

### Running the Demo

```bash
python main.py
```

This will run a comprehensive demonstration comparing all scheduling policies with sample workloads.

## 🔧 Configuration

### Cluster Configuration
- **Number of nodes**: Configurable via `num_nodes` parameter
- **GPUs per node**: Configurable via `gpus_per_node` parameter
- **Node resources**: CPU cores, memory, network capacity

### Job Configuration
- **GPU requirements**: Number of GPUs needed
- **Model information**: Deep learning model name and parameters
- **Execution parameters**: Duration, iterations, intervals

## 📊 Performance Metrics

### Key Performance Indicators

1. **Job Completion Time (JCT)**: Total time from submission to completion
2. **GPU Utilization**: Percentage of GPUs actively used
3. **Resource Fragmentation**: Measure of inefficient resource allocation
4. **Throughput**: Number of jobs completed per time unit

### Metric Collection

The system automatically collects:
- Real-time resource utilization
- Job execution statistics
- Cluster performance metrics
- Resource allocation patterns

## 🎯 Design Objectives

### Primary Goals
- **Minimize cluster-wide job completion time**
- **Maximize resource utilization**
- **Reduce resource fragmentation**

### Challenges Addressed

#### Challenge 1: Scheduling
- **Unpredictable training time**: Handled by adaptive scheduling policies
- **Job execution time optimization**: Multiple scheduling algorithms
- **Smooth loss curve utilization**: Time-based scheduling strategies

#### Challenge 2: Placement
- **Over-aggressive consolidation**: Balanced placement schemes
- **Queueing delay**: Intelligent resource allocation

## 🔬 Extending the System

### Adding New Schedulers

```python
class CustomScheduler(Scheduler):
    def __init__(self):
        super().__init__("Custom")
    
    def select_job(self, pending_jobs: List[Job]) -> Optional[Job]:
        # Implement custom selection logic
        pass

# Register with cluster
cluster.schedulers['custom'] = CustomScheduler()
```

### Adding New Placement Schemes

```python
class CustomPlacement(PlacementScheme):
    def __init__(self):
        super().__init__("Custom")
    
    def place_job(self, job: Job, nodes: List[_Node]) -> Optional[Tuple[_Node, List[int]]]:
        # Implement custom placement logic
        pass

# Register with cluster
cluster.placement_schemes['custom'] = CustomPlacement()
```

### Adding New Metrics

```python
# In MetricsCollector class
def record_custom_metric(self, metric_name: str, value: float):
    self.custom_metrics[metric_name].append(value)
```

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

## 📈 Future Enhancements

- **Real-time GPU monitoring integration**
- **Advanced placement algorithms** (bin-packing, genetic algorithms)
- **Multi-objective optimization** (JCT vs. energy efficiency)
- **Dynamic resource scaling**
- **Web-based dashboard**
- **Integration with Kubernetes/Docker**

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request


## 🙏 Acknowledgments

- Inspired by real-world GPU cluster management challenges
- Built for educational and research purposes
- Designed to be extensible for production use

---

**Note**: This system is designed for simulation and research purposes. For production GPU cluster management, consider integrating with existing solutions like Kubernetes, Slurm, or specialized GPU orchestration tools. # orca
