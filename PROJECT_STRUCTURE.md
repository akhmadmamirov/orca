# GPU Cluster Management System - Project Structure

## 📁 File Organization

```
Orca/
├── main.py                    # Main entry point with demo functionality
├── test_system.py            # Comprehensive test suite
├── example_usage.py          # Usage examples and demonstrations
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── PROJECT_STRUCTURE.md      # This file
│
├── models/                   # Data models and structures
│   ├── __init__.py          # Models package initialization
│   ├── job.py               # Job class and JobState enum
│   └── node.py              # Node class for GPU management
│
├── schedulers/               # Job scheduling policies
│   ├── __init__.py          # Schedulers package initialization
│   ├── base.py              # Base Scheduler abstract class
│   ├── fifo.py              # First-in-first-out scheduler
│   ├── sjf.py               # Smallest job first scheduler
│   ├── shortest.py          # Shortest remaining time scheduler
│   └── shortest_gpu.py      # Shortest remaining GPU time scheduler
│
├── placement/                # Job placement strategies
│   ├── __init__.py          # Placement package initialization
│   ├── base.py              # Base PlacementScheme abstract class
│   ├── first_fit.py         # First-fit placement scheme
│   └── best_fit.py          # Best-fit placement scheme
│
├── metrics/                  # Performance measurement
│   ├── __init__.py          # Metrics package initialization
│   └── collector.py         # MetricsCollector class
│
└── core/                    # Core system functionality
    ├── __init__.py          # Core package initialization
    └── cluster_manager.py   # Main GPUClusterManager class
```

## 🔧 Core Components

### `main.py` - Main Entry Point
**Purpose**: Entry point with demo functionality and system demonstration

**Features**:
- System demonstration with different schedulers
- Performance comparison across scheduling policies
- Sample workload execution

### `models/` - Data Models Package
**Purpose**: Contains all data structures and models

#### `models/job.py`
- **Job**: Represents deep learning jobs with states and metrics
- **JobState**: Enum for job lifecycle states (ADDED, PENDING, RUNNING, END, ERROR)

#### `models/node.py`
- **Node**: Manages GPU allocation and resource tracking
- GPU allocation/deallocation methods
- Resource utilization properties

### `schedulers/` - Scheduling Policies Package
**Purpose**: Implements various job scheduling algorithms

#### `schedulers/base.py`
- **Scheduler**: Abstract base class for all schedulers

#### `schedulers/fifo.py`
- **FIFOScheduler**: First-in-first-out scheduling

#### `schedulers/sjf.py`
- **SJFScheduler**: Smallest job first (by GPU count)

#### `schedulers/shortest.py`
- **ShortestScheduler**: Shortest remaining time first

#### `schedulers/shortest_gpu.py`
- **ShortestGPUScheduler**: Shortest remaining GPU time first

### `placement/` - Placement Strategies Package
**Purpose**: Implements GPU allocation and placement schemes

#### `placement/base.py`
- **PlacementScheme**: Abstract base class for placement strategies

#### `placement/first_fit.py`
- **FirstFitPlacement**: Allocate to first available node

#### `placement/best_fit.py`
- **BestFitPlacement**: Minimize resource fragmentation

### `metrics/` - Performance Measurement Package
**Purpose**: Collects and analyzes system performance metrics

#### `metrics/collector.py`
- **MetricsCollector**: Tracks job, cluster, and resource metrics
- Real-time performance monitoring
- Historical data collection and analysis

### `core/` - Core System Package
**Purpose**: Main orchestration and cluster management logic

#### `core/cluster_manager.py`
- **GPUClusterManager**: Main system orchestrator
- Job lifecycle management
- Resource allocation coordination
- Simulation engine

## 🏗️ System Architecture

### Package Dependencies
```
main.py
  └── core/
      ├── models/
      ├── schedulers/
      ├── placement/
      └── metrics/
```

### Data Flow
```
User Input → Job Submission → Queue Management → Scheduling → Placement → Execution → Metrics Collection
```

### Component Interactions
1. **Models**: Provide data structures for jobs, nodes, and system state
2. **Schedulers**: Select jobs based on policy
3. **Placement**: Map jobs to available resources
4. **Metrics**: Collect and analyze performance data
5. **Core**: Orchestrate all components and manage simulation

## 🎯 Key Features

### Modular Design
- **Separation of Concerns**: Each package has a specific responsibility
- **Extensibility**: Easy to add new schedulers, placement schemes, or metrics
- **Maintainability**: Clear organization makes code easier to understand and modify
- **Testability**: Individual components can be tested in isolation

### Job Representation
- **States**: Complete lifecycle management (ADDED → PENDING → RUNNING → END)
- **Properties**: GPU requirements, model info, execution parameters
- **Tracking**: Execution time, pending time, resource allocation

### Resource Management
- **GPU Allocation**: Dynamic allocation and deallocation
- **Node Management**: Utilization tracking and status monitoring
- **Resource Fragmentation**: Measurement and optimization

### Performance Metrics
- **Job-level**: Completion time, execution efficiency
- **Cluster-level**: Overall utilization and throughput
- **Resource-level**: Per-node performance characteristics

## 🔬 Extensibility

### Adding New Schedulers
```python
# Create new scheduler file: schedulers/custom.py
from schedulers.base import Scheduler
from models.job import Job

class CustomScheduler(Scheduler):
    def select_job(self, pending_jobs: List[Job]) -> Optional[Job]:
        # Custom selection logic
        pass

# Update schedulers/__init__.py to include new scheduler
```

### Adding New Placement Schemes
```python
# Create new placement file: placement/custom.py
from placement.base import PlacementScheme
from models.job import Job
from models.node import Node

class CustomPlacement(PlacementScheme):
    def place_job(self, job: Job, nodes: List[Node]):
        # Custom placement logic
        pass

# Update placement/__init__.py to include new scheme
```

### Adding New Metrics
```python
# Extend metrics/collector.py
class MetricsCollector:
    def record_custom_metric(self, metric_name: str, value: float):
        self.custom_metrics[metric_name].append(value)
```

## 🚀 Getting Started

### Quick Start
1. **Run Demo**: `python3 main.py`
2. **Run Tests**: `python3 test_system.py`
3. **Run Examples**: `python3 example_usage.py`

### Custom Usage
```python
from core import GPUClusterManager
from models import Job, JobState

# Create cluster
cluster = GPUClusterManager(num_nodes=4, gpus_per_node=4)

# Submit jobs
cluster.submit_job(num_gpu=2, model_name="ResNet50", duration=30)

# Run simulation
for step in range(50):
    cluster.update_simulation(1.0)

# Check results
cluster.print_status()
```

## 📊 Benefits of Refactoring

### Code Organization
- **Logical Separation**: Related functionality grouped together
- **Clear Dependencies**: Explicit import relationships
- **Easier Navigation**: Developers can quickly find relevant code

### Maintainability
- **Single Responsibility**: Each class has one clear purpose
- **Reduced Coupling**: Components depend on abstractions, not implementations
- **Easier Debugging**: Issues can be isolated to specific modules

### Testing
- **Unit Testing**: Individual components can be tested independently
- **Mocking**: Dependencies can be easily mocked for testing
- **Integration Testing**: Clear interfaces make integration testing straightforward

### Collaboration
- **Parallel Development**: Multiple developers can work on different modules
- **Code Reviews**: Smaller, focused files are easier to review
- **Documentation**: Each module can have focused documentation

## 🔮 Future Enhancements

### Planned Features
- **Real-time GPU monitoring integration**
- **Advanced placement algorithms** (bin-packing, genetic algorithms)
- **Multi-objective optimization** (JCT vs. energy efficiency)
- **Dynamic resource scaling**
- **Web-based dashboard**
- **Kubernetes integration**

### Research Applications
- **Scheduling policy research**
- **Resource allocation optimization**
- **Performance benchmarking**
- **Workload characterization**

---

**Note**: This refactored system maintains all original functionality while providing a clean, modular architecture that's easier to maintain, extend, and test. The separation of concerns makes it ideal for research and educational purposes. 