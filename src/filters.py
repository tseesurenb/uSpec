"""
Filter Patterns for Universal Spectral Collaborative Filtering
Comprehensive collection of digital filter coefficients for various filtering applications

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
Created: June 4, 2025
"""

import torch
import numpy as np

# Main filter patterns dictionary
filter_patterns = {
    # Low-pass filters (Smoothing filters)
    'butterworth': [1.0, -0.6, 0.2, -0.05, 0.01, -0.002, 0.0003, -0.00005],
    'chebyshev': [1.0, -0.4, 0.1, -0.01, 0.001, -0.0001, 0.00001, -0.000001],
    'smooth': [1.0, -0.5, 0.1, -0.02, 0.004, -0.0008, 0.00015, -0.00003],
    'bessel': [1.0, -0.3, 0.06, -0.008, 0.0008, -0.00006, 0.000004, -0.0000002],
    'gaussian': [1.0, -0.7, 0.15, -0.03, 0.005, -0.0007, 0.00008, -0.000008],
    'conservative': [1.0, -0.2, 0.03, -0.002, 0.0001, -0.000005, 0.0000002, -0.00000001],
    'aggressive': [1.0, -0.8, 0.3, -0.08, 0.015, -0.002, 0.0002, -0.00002],
    'elliptic_lp': [1.0, -0.5, 0.12, -0.018, 0.002, -0.0002, 0.00002, -0.000002],
    'hamming_lp': [1.0, -0.45, 0.08, -0.012, 0.0015, -0.00015, 0.000012, -0.000001],
    
    # High-pass filters (Detail preservation)
    'butterworth_hp': [0.0001, 0.002, -0.01, 0.05, -0.2, 0.6, -1.0, 1.0],
    'chebyshev_hp': [0.000001, 0.00001, -0.0001, 0.001, -0.01, 0.1, -0.4, 1.0],
    'bessel_hp': [0.0000002, 0.000004, -0.00006, 0.0008, -0.008, 0.06, -0.3, 1.0],
    'elliptic_hp': [0.000002, 0.00002, -0.0002, 0.002, -0.018, 0.12, -0.5, 1.0],
    'aggressive_hp': [0.00002, 0.0002, -0.002, 0.015, -0.08, 0.3, -0.8, 1.0],
    'gentle_hp': [0.00000001, 0.0000002, -0.000005, 0.0001, -0.002, 0.03, -0.2, 1.0],
    
    # Band-pass filters (Mid-frequency emphasis)
    'butterworth_bp': [0.001, -0.05, 0.3, -0.7, 0.8, -0.4, 0.1, -0.01],
    'chebyshev_bp': [0.002, -0.08, 0.25, -0.6, 0.75, -0.35, 0.08, -0.008],
    'bessel_bp': [0.0005, -0.03, 0.2, -0.5, 0.6, -0.3, 0.06, -0.005],
    'gaussian_bp': [0.003, -0.12, 0.4, -0.8, 0.9, -0.45, 0.12, -0.015],
    'narrow_bp': [0.0001, -0.01, 0.15, -0.4, 0.5, -0.25, 0.05, -0.003],
    'wide_bp': [0.01, -0.2, 0.6, -1.2, 1.3, -0.65, 0.2, -0.03],
    
    # Band-stop (notch) filters (Frequency removal)
    'butterworth_bs': [1.0, -0.2, -0.3, 0.7, -0.8, 0.4, -0.1, 0.01],
    'chebyshev_bs': [1.0, -0.15, -0.25, 0.6, -0.75, 0.35, -0.08, 0.008],
    'bessel_bs': [1.0, -0.1, -0.2, 0.5, -0.6, 0.3, -0.06, 0.005],
    'notch_sharp': [1.0, -0.05, -0.4, 0.9, -1.0, 0.5, -0.12, 0.015],
    'notch_wide': [1.0, -0.3, -0.1, 0.4, -0.5, 0.25, -0.05, 0.003],
    
    # All-pass filters (Phase modification)
    'allpass_linear': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'allpass_delay': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'allpass_phase1': [1.0, 0.2, -0.15, 0.1, -0.05, 0.02, -0.01, 0.005],
    'allpass_phase2': [1.0, -0.3, 0.2, -0.1, 0.05, -0.02, 0.01, -0.005],
    'allpass_dispersive': [1.0, 0.1, -0.05, 0.02, -0.01, 0.005, -0.002, 0.001],
    
    # Adaptive and specialized filters
    'adaptive_smooth': [1.0, -0.4, 0.08, -0.008, 0.0006, -0.00003, 0.000001, 0.0],
    'adaptive_sharp': [1.0, -0.7, 0.25, -0.05, 0.008, -0.001, 0.0001, -0.00001],
    'wiener_like': [1.0, -0.35, 0.06, -0.005, 0.0003, -0.00001, 0.0000003, 0.0],
    'kalman_like': [1.0, -0.55, 0.12, -0.015, 0.0012, -0.00007, 0.000003, 0.0],
    'median_like': [1.0, -0.25, 0.04, -0.003, 0.0001, -0.000003, 0.0000001, 0.0],
    
    # Frequency-specific filters
    'low_freq_enhance': [1.2, -0.8, 0.3, -0.08, 0.015, -0.002, 0.0002, -0.00002],
    'mid_freq_enhance': [0.8, -0.2, 0.6, -0.4, 0.15, -0.03, 0.004, -0.0003],
    'high_freq_enhance': [0.2, -0.1, 0.3, -0.5, 0.7, -0.6, 0.4, -0.2],
    'multi_band': [1.0, -0.3, 0.2, -0.4, 0.3, -0.2, 0.1, -0.05],
    
    # Multi-band variants (similar to multi_band)
    'multi_band_v2': [1.0, -0.4, 0.25, -0.35, 0.25, -0.15, 0.08, -0.04],
    'multi_band_v3': [1.0, -0.25, 0.15, -0.45, 0.35, -0.25, 0.12, -0.06],
    'multi_band_aggressive': [1.0, -0.5, 0.3, -0.6, 0.4, -0.3, 0.15, -0.08],
    'multi_band_gentle': [1.0, -0.2, 0.1, -0.3, 0.2, -0.1, 0.05, -0.02],
    'multi_band_balanced': [1.0, -0.35, 0.22, -0.38, 0.28, -0.18, 0.09, -0.045],
    'multi_band_wide': [1.0, -0.45, 0.35, -0.25, 0.15, -0.25, 0.2, -0.1],
    'multi_band_narrow': [1.0, -0.15, 0.08, -0.5, 0.4, -0.15, 0.06, -0.03],
    
    # Oscillatory patterns (similar alternating characteristics)
    'oscillatory_decay': [1.0, -0.6, 0.36, -0.216, 0.1296, -0.078, 0.047, -0.028],
    'oscillatory_moderate': [1.0, -0.4, 0.16, -0.32, 0.256, -0.128, 0.064, -0.032],
    'oscillatory_strong': [1.0, -0.7, 0.49, -0.343, 0.24, -0.168, 0.118, -0.083],
    'oscillatory_gentle': [1.0, -0.3, 0.09, -0.18, 0.144, -0.072, 0.036, -0.018],
    
    # Harmonic series (musical/natural patterns)
    'harmonic_fundamental': [1.0, -0.5, 0.33, -0.25, 0.2, -0.167, 0.143, -0.125],
    'harmonic_rich': [1.0, -0.33, 0.2, -0.5, 0.25, -0.167, 0.143, -0.111],
    'harmonic_sparse': [1.0, -0.25, 0.125, -0.333, 0.167, -0.1, 0.071, -0.063],
    
    # Fibonacci-inspired (natural growth patterns)
    'fibonacci_alt': [1.0, -0.618, 0.382, -0.236, 0.146, -0.09, 0.056, -0.034],
    'fibonacci_decay': [1.0, -0.382, 0.146, -0.236, 0.09, -0.146, 0.056, -0.034],
    
    # Resonance patterns (emphasize specific frequency relationships)
    'resonance_1_3': [1.0, -0.2, 0.6, -0.1, 0.4, -0.05, 0.2, -0.02],
    'resonance_2_5': [1.0, -0.4, 0.1, -0.6, 0.05, -0.4, 0.02, -0.2],
    'resonance_balanced': [1.0, -0.3, 0.4, -0.2, 0.3, -0.15, 0.2, -0.1],
    
    # Spectral envelope patterns
    'envelope_triangle': [1.0, -0.4, 0.6, -0.8, 0.6, -0.4, 0.2, -0.1],
    'envelope_trapezoid': [1.0, -0.3, 0.5, -0.5, 0.5, -0.5, 0.3, -0.1],
    'envelope_gaussian': [1.0, -0.35, 0.45, -0.55, 0.45, -0.35, 0.25, -0.15],
    
    # Edge detection and derivative filters
    'edge_detect': [0.0, 0.5, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
    'gradient_x': [-0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
    'laplacian': [0.0, 1.0, -2.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    'sobel_like': [-0.25, 0.0, 0.25, -0.5, 0.0, 0.5, -0.25, 0.25],
    
    # Noise reduction filters
    'denoise_gentle': [1.0, -0.1, 0.01, -0.001, 0.0001, 0.0, 0.0, 0.0],
    'denoise_moderate': [1.0, -0.3, 0.05, -0.005, 0.0003, -0.00001, 0.0, 0.0],
    'denoise_strong': [1.0, -0.6, 0.15, -0.025, 0.003, -0.0002, 0.00001, 0.0],
    'bilateral_like': [1.0, -0.4, 0.08, -0.008, 0.0004, -0.00001, 0.0, 0.0],
    
    # Identity and special cases
    'identity': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'unity': [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
    'alternating': [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
    'exponential_decay': [1.0, -0.5, 0.25, -0.125, 0.0625, -0.03125, 0.015625, -0.0078125]
}

# Filter categories for easy reference
filter_categories = {
    'lowpass': ['butterworth', 'chebyshev', 'smooth', 'bessel', 'gaussian', 'conservative', 
                'aggressive', 'elliptic_lp', 'hamming_lp'],
    'highpass': ['butterworth_hp', 'chebyshev_hp', 'bessel_hp', 'elliptic_hp', 
                 'aggressive_hp', 'gentle_hp'],
    'bandpass': ['butterworth_bp', 'chebyshev_bp', 'bessel_bp', 'gaussian_bp', 
                 'narrow_bp', 'wide_bp'],
    'bandstop': ['butterworth_bs', 'chebyshev_bs', 'bessel_bs', 'notch_sharp', 'notch_wide'],
    'allpass': ['allpass_linear', 'allpass_delay', 'allpass_phase1', 'allpass_phase2', 
                'allpass_dispersive'],
    'adaptive': ['adaptive_smooth', 'adaptive_sharp', 'wiener_like', 'kalman_like', 'median_like'],
    'frequency_specific': ['low_freq_enhance', 'mid_freq_enhance', 'high_freq_enhance', 'multi_band',
                           'multi_band_v2', 'multi_band_v3', 'multi_band_aggressive', 'multi_band_gentle',
                           'multi_band_balanced', 'multi_band_wide', 'multi_band_narrow', 'oscillatory_decay',
                           'oscillatory_moderate', 'oscillatory_strong', 'oscillatory_gentle', 'harmonic_fundamental',
                           'harmonic_rich', 'harmonic_sparse', 'fibonacci_alt', 'fibonacci_decay',
                           'resonance_1_3', 'resonance_2_5', 'resonance_balanced', 'envelope_triangle',
                           'envelope_trapezoid', 'envelope_gaussian'],
    'edge_detection': ['edge_detect', 'gradient_x', 'laplacian', 'sobel_like'],
    'noise_reduction': ['denoise_gentle', 'denoise_moderate', 'denoise_strong', 'bilateral_like'],
    'special': ['identity', 'unity', 'alternating', 'exponential_decay']
}

def get_filter_coefficients(filter_name, order=None, as_tensor=False):
    """
    Get filter coefficients by name.
    
    Args:
        filter_name (str): Name of the filter pattern
        order (int, optional): If specified, truncate or pad coefficients to this order
        as_tensor (bool): If True, return as PyTorch tensor, else return as list
    
    Returns:
        list or torch.Tensor: Filter coefficients
    """
    if filter_name not in filter_patterns:
        raise ValueError(f"Unknown filter pattern: {filter_name}. Available: {list(filter_patterns.keys())}")
    
    coeffs = filter_patterns[filter_name].copy()
    
    if order is not None:
        if len(coeffs) > order + 1:
            coeffs = coeffs[:order + 1]
        elif len(coeffs) < order + 1:
            coeffs.extend([0.0] * (order + 1 - len(coeffs)))
    
    if as_tensor:
        return torch.tensor(coeffs, dtype=torch.float32)
    
    return coeffs

def list_filters_by_category(category=None):
    """
    List available filters, optionally by category.
    
    Args:
        category (str, optional): Filter category ('lowpass', 'highpass', etc.)
    
    Returns:
        list: Filter names
    """
    if category is None:
        return list(filter_patterns.keys())
    
    if category not in filter_categories:
        raise ValueError(f"Unknown category: {category}. Available: {list(filter_categories.keys())}")
    
    return filter_categories[category]

def get_filter_info(filter_name):
    """
    Get information about a specific filter.
    
    Args:
        filter_name (str): Name of the filter pattern
    
    Returns:
        dict: Filter information including coefficients, category, and properties
    """
    if filter_name not in filter_patterns:
        raise ValueError(f"Unknown filter pattern: {filter_name}")
    
    # Find which category this filter belongs to
    category = None
    for cat, filters in filter_categories.items():
        if filter_name in filters:
            category = cat
            break
    
    coeffs = filter_patterns[filter_name]
    
    return {
        'name': filter_name,
        'category': category,
        'coefficients': coeffs,
        'order': len(coeffs) - 1,
        'num_coefficients': len(coeffs),
        'max_abs_coeff': max(abs(c) for c in coeffs),
        'sum_coeffs': sum(coeffs)
    }

def print_filter_summary():
    """Print a summary of all available filters."""
    print("Available Filter Patterns:")
    print("=" * 50)
    
    for category, filters in filter_categories.items():
        print(f"\n{category.upper()} FILTERS:")
        for filter_name in filters:
            info = get_filter_info(filter_name)
            print(f"  {filter_name:20} | Order: {info['order']:2} | Sum: {info['sum_coeffs']:8.4f}")

# Example usage function
def example_usage():
    """Demonstrate how to use the filter patterns."""
    print("Filter Patterns Module - Usage Examples")
    print("=" * 40)
    
    # Get a specific filter
    butterworth_coeffs = get_filter_coefficients('butterworth', as_tensor=True)
    print(f"Butterworth coefficients: {butterworth_coeffs}")
    
    # Get filters by category
    lowpass_filters = list_filters_by_category('lowpass')
    print(f"Lowpass filters: {lowpass_filters}")
    
    # Get filter information
    info = get_filter_info('gaussian')
    print(f"Gaussian filter info: {info}")
    
    # Print summary
    print_filter_summary()

