"""
Inference utility functions

This module provides essential utility functions supporting the TSGBench standardized
preprocessing pipeline.

Key Features:
- Compressed data I/O using mgzip for efficient storage
- Device management for GPU/CPU computation
"""


def read_mgzip_data(path):
    """
    Read compressed pickle data from a file using mgzip.
    
    Args:
        path (str): Path to the compressed pickle file
        
    Returns:
        Any: The unpickled data object
    """
    with mgzip.open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def write_mgzip_data(content, path):
    """
    Write data to a compressed pickle file using mgzip.
    
    Args:
        content (Any): Data to serialize and save
        path (str): Path where to save the compressed file
    
    Note:
        TODO: determine the output path for evals
    """
    # make_sure_path_exist(path)
    # with mgzip.open(path, 'wb') as f:
    #     pickle.dump(content, f)

def write_json_data(content, path):
    """
    Write data to a JSON file with proper formatting.
    
    Args:
        content (dict): Dictionary to save as JSON
        path (str): Path where to save the JSON file
        
    Note:
        TODO: determine the output path for evals
    """
    pass
    # make_sure_path_exist(path)
    # with open('data.json', 'w') as json_file:
    #     json.dump(content, json_file, indent=4)

def determine_device(no_cuda, cuda_device):
    """
    Determine the appropriate computing device (CPU or GPU) for PyTorch operations.
    
    This function automatically selects the best available device based on CUDA availability
    and user preferences. It handles both single and multi-GPU systems.
    
    Args:
        no_cuda (bool): If True, force CPU usage regardless of GPU availability
        cuda_device (int): Specific CUDA device index to use (for multi-GPU systems)
        
    Returns:
        torch.device: The selected device for computation

    Note:
        - If no_cuda=True, always returns CPU device
        - If CUDA is not available, falls back to CPU
        - For multi-GPU systems, uses the specified cuda_device
        - For single GPU systems, uses cuda:0
    """
    # Determine device (cpu/gpu)
    if no_cuda or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        if torch.cuda.device_count()>1:
            device = torch.device('cuda', cuda_device)
        else:
            device = torch.device('cuda', 0)
    return device