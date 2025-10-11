"""
Performance measurement utility functions.

This module provides utilities for measuring runtime and memory usage.
"""

import time
import psutil
import os
import torch


def measure_runtime(generate_func, *args, **kwargs):
    """
    Measure runtime (in seconds) for generating synthetic data.
    
    Args:
        generate_func (callable): Function to generate synthetic data
        *args: Variable length argument list passed to generate_func
        **kwargs: Arbitrary keyword arguments passed to generate_func
    
    Returns:
        float: Time taken to generate in seconds
    """
    start_time = time.perf_counter()
    _ = generate_func(*args, **kwargs)
    end_time = time.perf_counter()
    return round(end_time - start_time, 4)


def measure_peak_memory(generate_func, *args, **kwargs):
    """
    Measure peak memory usage (CPU + GPU) during synthetic data generation.
    
    Args:
        generate_func (callable): Function to generate synthetic data
        *args: Variable length argument list passed to generate_func
        **kwargs: Arbitrary keyword arguments passed to generate_func
    
    Returns: 
        float: Peak memory usage in MB
    """
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / 1e6  # MB

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    _ = generate_func(*args, **kwargs)

    end_mem = process.memory_info().rss / 1e6
    cpu_mem = max(0, end_mem - start_mem)

    gpu_mem = 0
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.max_memory_allocated() / 1e6

    return round(cpu_mem + gpu_mem, 2)
