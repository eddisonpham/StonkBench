"""
Measure runtime (in seconds) for generating synthetic data.
"""
import time


def measure_runtime(generate_func, *args, **kwargs):
    """
    Measure runtime (in seconds) for generating synthetic data.
    
    Args:
        generate_func (callable): Function to generate synthetic data
            - *args, **kwargs: arguments passed to generate_func
    
    Returns:
        float: Time taken to generate in seconds
    """
    start_time = time.perf_counter()
    _ = generate_func(*args, **kwargs)
    end_time = time.perf_counter()
    return round(end_time - start_time, 4)
