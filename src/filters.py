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
    
    # Soft oscillatory variants (based on oscillatory_soft success)
    'oscillatory_soft': [1.0, -0.35, 0.1225, -0.21, 0.168, -0.084, 0.042, -0.021],
    'oscillatory_soft_v2': [1.0, -0.36, 0.1296, -0.216, 0.1728, -0.0864, 0.0432, -0.0216],
    'oscillatory_soft_v3': [1.0, -0.34, 0.1156, -0.204, 0.1632, -0.0816, 0.0408, -0.0204],
    'oscillatory_soft_plus': [1.0, -0.37, 0.1369, -0.222, 0.1776, -0.0888, 0.0444, -0.0222],
    'oscillatory_soft_minus': [1.0, -0.33, 0.1089, -0.198, 0.1584, -0.0792, 0.0396, -0.0198],
    
    # Fine-tuned around the 0.35 coefficient
    'soft_tuned_351': [1.0, -0.351, 0.123201, -0.2106, 0.16848, -0.08424, 0.04212, -0.02106],
    'soft_tuned_352': [1.0, -0.352, 0.123904, -0.2112, 0.16896, -0.08448, 0.04224, -0.02112],
    'soft_tuned_353': [1.0, -0.353, 0.124609, -0.2118, 0.16944, -0.08472, 0.04236, -0.02118],
    'soft_tuned_348': [1.0, -0.348, 0.121104, -0.2088, 0.16704, -0.08352, 0.04176, -0.02088],
    'soft_tuned_349': [1.0, -0.349, 0.121801, -0.2094, 0.16752, -0.08376, 0.04188, -0.02094],
    
    # Golden ratio and natural mathematics variants (based on soft_golden_ratio success)
    'soft_golden_ratio': [1.0, -0.35, 0.1225, -0.214, 0.1519, -0.0855, 0.0532, -0.0329],
    'golden_ratio_pure': [1.0, -0.618, 0.382, -0.236, 0.146, -0.09, 0.056, -0.034],
    'golden_ratio_soft_v2': [1.0, -0.35, 0.1225, -0.217, 0.1526, -0.0861, 0.0535, -0.0331],
    'golden_ratio_soft_v3': [1.0, -0.35, 0.1225, -0.211, 0.1512, -0.0849, 0.0529, -0.0327],
    'golden_ratio_balanced': [1.0, -0.35, 0.1225, -0.215, 0.152, -0.0857, 0.0533, -0.033],
    
    # Fibonacci sequence variations
    'fibonacci_soft': [1.0, -0.35, 0.1225, -0.213, 0.1597, -0.0987, 0.061, -0.0377],
    'fibonacci_gentle': [1.0, -0.35, 0.1225, -0.212, 0.1584, -0.0979, 0.0605, -0.0374],
    'fibonacci_precise': [1.0, -0.35, 0.1225, -0.2135, 0.1597, -0.0987, 0.061, -0.0377],
    
    # Natural constants (e, π, φ) inspired
    'euler_soft': [1.0, -0.35, 0.1225, -0.2148, 0.1445, -0.0872, 0.0531, -0.0324],
    'pi_ratio_soft': [1.0, -0.35, 0.1225, -0.2146, 0.1592, -0.1019, 0.0648, -0.0414],
    'natural_harmony': [1.0, -0.35, 0.1225, -0.2142, 0.1528, -0.0866, 0.0541, -0.0334],
    
    # Golden ratio with different base coefficients
    'golden_034': [1.0, -0.34, 0.1156, -0.208, 0.1473, -0.0829, 0.0516, -0.0319],
    'golden_036': [1.0, -0.36, 0.1296, -0.220, 0.1564, -0.088, 0.0548, -0.0339],
    'golden_348': [1.0, -0.348, 0.121104, -0.2115, 0.1502, -0.0846, 0.0526, -0.0325],
    'golden_352': [1.0, -0.352, 0.123904, -0.2125, 0.1511, -0.0851, 0.0529, -0.0327],
    
    # Mathematical series combinations
    'golden_fibonacci_hybrid': [1.0, -0.35, 0.1225, -0.2142, 0.1597, -0.0901, 0.0566, -0.0351],
    'golden_harmonic': [1.0, -0.35, 0.1225, -0.2145, 0.1458, -0.0875, 0.0583, -0.0389],
    'golden_geometric': [1.0, -0.35, 0.1225, -0.2141, 0.1497, -0.0859, 0.0531, -0.0322],
    
    # Sacred geometry inspired (phi, golden angle, etc.)
    'sacred_ratio_1': [1.0, -0.35, 0.1225, -0.2144, 0.1539, -0.0888, 0.0549, -0.0340],
    'sacred_ratio_2': [1.0, -0.35, 0.1225, -0.2138, 0.1499, -0.0823, 0.0515, -0.0317],
    'golden_spiral': [1.0, -0.35, 0.1225, -0.2143, 0.1618, -0.0955, 0.0618, -0.0382],
    
    # Nature-inspired mathematical patterns
    'nautilus_pattern': [1.0, -0.35, 0.1225, -0.2142, 0.1534, -0.0947, 0.0585, -0.0361],
    'sunflower_spiral': [1.0, -0.35, 0.1225, -0.2147, 0.1472, -0.0859, 0.0574, -0.0347],
    'pine_cone_ratio': [1.0, -0.35, 0.1225, -0.2140, 0.1556, -0.0901, 0.0542, -0.0334],
    
    # Musical harmony ratios
    'perfect_fifth': [1.0, -0.35, 0.1225, -0.213, 0.1472, -0.0884, 0.0531, -0.0354],
    'golden_fourth': [1.0, -0.35, 0.1225, -0.2146, 0.1562, -0.0937, 0.0562, -0.0337],
    'harmonic_series': [1.0, -0.35, 0.1225, -0.2142, 0.1519, -0.0844, 0.0563, -0.0352],
    
    # Fine-tuned golden variations
    'golden_optimized_1': [1.0, -0.35, 0.1225, -0.2142, 0.1519, -0.0856, 0.0533, -0.033],
    'golden_optimized_2': [1.0, -0.35, 0.1225, -0.2141, 0.1518, -0.0854, 0.0531, -0.0328],
    'golden_optimized_3': [1.0, -0.35, 0.1225, -0.2143, 0.152, -0.0857, 0.0534, -0.0331],
    
    # Structured decay variants
    'soft_structured_1': [1.0, -0.35, 0.12, -0.21, 0.17, -0.085, 0.043, -0.021],
    'soft_structured_2': [1.0, -0.35, 0.125, -0.21, 0.165, -0.083, 0.041, -0.0205],
    'soft_structured_3': [1.0, -0.35, 0.123, -0.21, 0.167, -0.084, 0.042, -0.021],
    
    # Precision variants (small adjustments)
    'soft_precise_1': [1.0, -0.35, 0.1225, -0.21, 0.168, -0.0835, 0.0418, -0.0209],
    'soft_precise_2': [1.0, -0.35, 0.1225, -0.21, 0.168, -0.0845, 0.0422, -0.0211],
    'soft_precise_3': [1.0, -0.35, 0.1225, -0.209, 0.168, -0.084, 0.042, -0.021],
    'soft_precise_4': [1.0, -0.35, 0.1225, -0.211, 0.168, -0.084, 0.042, -0.021],
    
    # Soft with different alternation patterns
    'soft_alt_strong': [1.0, -0.35, 0.1225, -0.22, 0.168, -0.084, 0.042, -0.021],
    'soft_alt_weak': [1.0, -0.35, 0.1225, -0.20, 0.168, -0.084, 0.042, -0.021],
    'soft_alt_balanced': [1.0, -0.35, 0.1225, -0.2125, 0.168, -0.084, 0.042, -0.021],
    
    # Soft with modified tail behavior
    'soft_long_tail': [1.0, -0.35, 0.1225, -0.21, 0.168, -0.084, 0.045, -0.024],
    'soft_short_tail': [1.0, -0.35, 0.1225, -0.21, 0.168, -0.084, 0.039, -0.018],
    'soft_extended': [1.0, -0.35, 0.1225, -0.21, 0.168, -0.084, 0.042, -0.0215],
    
    # Soft with ratio modifications
    'soft_ratio_08': [1.0, -0.35, 0.1225, -0.21, 0.168, -0.0672, 0.0336, -0.0168],
    'soft_ratio_12': [1.0, -0.35, 0.1225, -0.21, 0.168, -0.1008, 0.0504, -0.0252],
    'soft_ratio_09': [1.0, -0.35, 0.1225, -0.21, 0.168, -0.0756, 0.0378, -0.0189],
    
    # Gentle patterns with different decay rates
    'gentle_linear_decay': [1.0, -0.3, 0.06, -0.15, 0.12, -0.09, 0.06, -0.03],
    'gentle_sqrt_decay': [1.0, -0.3, 0.1342, -0.1732, 0.1342, -0.1095, 0.0894, -0.0775],
    'gentle_exp_decay': [1.0, -0.3, 0.09, -0.162, 0.1296, -0.0778, 0.0467, -0.028],
    'gentle_cubic_decay': [1.0, -0.3, 0.027, -0.135, 0.1215, -0.0729, 0.0437, -0.0262],
    
    # Fibonacci-ratio gentle patterns
    'gentle_fibonacci': [1.0, -0.309, 0.0955, -0.191, 0.118, -0.073, 0.045, -0.028],
    'gentle_golden': [1.0, -0.318, 0.101, -0.196, 0.124, -0.079, 0.05, -0.032],
    
    # Wave-like gentle patterns
    'gentle_sine_like': [1.0, -0.3, 0.07, -0.17, 0.135, -0.068, 0.034, -0.017],
    'gentle_cosine_like': [1.0, -0.29, 0.1, -0.175, 0.14, -0.07, 0.035, -0.0175],
    'gentle_triangle': [1.0, -0.31, 0.093, -0.186, 0.149, -0.074, 0.037, -0.0186],
    
    # Damped oscillations (physics-inspired)
    'damped_oscillation_1': [1.0, -0.3, 0.08, -0.17, 0.13, -0.065, 0.032, -0.016],
    'damped_oscillation_2': [1.0, -0.31, 0.095, -0.185, 0.145, -0.075, 0.038, -0.019],
    'damped_oscillation_3': [1.0, -0.29, 0.085, -0.175, 0.135, -0.07, 0.035, -0.0175],
    
    # Controlled alternation patterns
    'controlled_alt_soft': [1.0, -0.27, 0.073, -0.162, 0.1296, -0.0648, 0.0324, -0.0162],
    'controlled_alt_medium': [1.0, -0.33, 0.109, -0.198, 0.1584, -0.0792, 0.0396, -0.0198],
    'controlled_alt_fine': [1.0, -0.285, 0.081, -0.171, 0.1368, -0.0684, 0.0342, -0.0171],
    
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
                           'oscillatory_moderate', 'oscillatory_strong', 'oscillatory_gentle', 
                           'oscillatory_ultra_gentle', 'oscillatory_micro', 'oscillatory_soft',
                           'oscillatory_soft_v2', 'oscillatory_soft_v3', 'oscillatory_soft_plus', 'oscillatory_soft_minus',
                           'soft_tuned_351', 'soft_tuned_352', 'soft_tuned_353', 'soft_tuned_348', 'soft_tuned_349',
                           'soft_golden_ratio', 'soft_sqrt_pattern', 'soft_cubic_pattern', 'soft_harmonic',
                           'soft_structured_1', 'soft_structured_2', 'soft_structured_3',
                           'soft_precise_1', 'soft_precise_2', 'soft_precise_3', 'soft_precise_4',
                           'soft_alt_strong', 'soft_alt_weak', 'soft_alt_balanced',
                           'soft_long_tail', 'soft_short_tail', 'soft_extended',
                           'soft_ratio_08', 'soft_ratio_12', 'soft_ratio_09',
                           'oscillatory_smooth', 'oscillatory_subtle', 'oscillatory_refined', 'oscillatory_calm',
                           'gentle_linear_decay', 'gentle_sqrt_decay', 'gentle_exp_decay', 'gentle_cubic_decay',
                           'gentle_fibonacci', 'gentle_golden', 'gentle_sine_like', 'gentle_cosine_like',
                           'gentle_triangle', 'damped_oscillation_1', 'damped_oscillation_2', 'damped_oscillation_3',
                           'controlled_alt_soft', 'controlled_alt_medium', 'controlled_alt_fine',
                           'harmonic_fundamental', 'harmonic_rich', 'harmonic_sparse', 'fibonacci_alt', 
                           'fibonacci_decay', 'resonance_1_3', 'resonance_2_5', 'resonance_balanced', 
                           'envelope_triangle', 'envelope_trapezoid', 'envelope_gaussian'],
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

