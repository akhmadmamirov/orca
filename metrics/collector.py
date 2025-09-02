"""
Metrics collection and analysis for GPU Cluster Management System
"""

from collections import defaultdict
import statistics


class MetricsCollector:
    """Collects and calculates system metrics"""
    
    def __init__(self):
        self.job_metrics = defaultdict(list)
        self.cluster_metrics = defaultdict(list)
        self.resource_metrics = defaultdict(list)
    
    def record_job_metric(self, job_id: str, metric_name: str, value: float):
        """Record a job-level metric"""
        self.job_metrics[f"{job_id}_{metric_name}"].append(value)
    
    def record_cluster_metric(self, metric_name: str, value: float):
        """Record a cluster-level metric"""
        self.cluster_metrics[metric_name].append(value)
    
    def record_resource_metric(self, node_id: str, metric_name: str, value: float):
        """Record a resource-level metric"""
        self.resource_metrics[f"{node_id}_{metric_name}"].append(value)
    
    def get_job_completion_time(self, job_id: str) -> float:
        """Get job completion time for a specific job"""
        key = f"{job_id}_completion_time"
        if key in self.job_metrics and self.job_metrics[key]:
            return self.job_metrics[key][-1]
        return 0.0
    
    def get_average_jct(self) -> float:
        """Get average job completion time"""
        jct_values = []
        for k, v in self.job_metrics.items():
            if 'completion_time' in k and v:
                jct_values.extend(v)
        return statistics.mean(jct_values) if jct_values else 0.0
    
    def get_gpu_utilization(self) -> float:
        """Get overall GPU utilization"""
        util_values = self.cluster_metrics.get('gpu_utilization', [])
        return statistics.mean(util_values) if util_values else 0.0
    
    def get_resource_fragmentation(self) -> float:
        """Get resource fragmentation metric"""
        frag_values = self.cluster_metrics.get('resource_fragmentation', [])
        return statistics.mean(frag_values) if frag_values else 0.0 