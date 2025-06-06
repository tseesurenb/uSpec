"""
Filter Patterns for Universal Spectral Collaborative Filtering
Focused on recommendation-relevant filters with enhanced capacity models

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
Created: June 4, 2025
"""

import torch
import torch.nn as nn
import numpy as np

# Recommendation-relevant filter patterns (cleaned up)
filter_patterns = {
    # Core smoothing filters (essential for CF)
    'smooth': [1.0, -0.5, 0.1, -0.02, 0.004, -0.0008, 0.00015, -0.00003],
    'butterworth': [1.0, -0.6, 0.2, -0.05, 0.01, -0.002, 0.0003, -0.00005],
    'gaussian': [1.0, -0.7, 0.15, -0.03, 0.005, -0.0007, 0.00008, -0.000008],
    'bessel': [1.0, -0.3, 0.06, -0.008, 0.0008, -0.00006, 0.000004, -0.0000002],
    'conservative': [1.0, -0.2, 0.03, -0.002, 0.0001, -0.000005, 0.0000002, -0.00000001],
    
    # Golden ratio variants (high-performance for CF)
    'golden_034': [1.0, -0.34, 0.1156, -0.208, 0.1473, -0.0829, 0.0516, -0.0319],
    'golden_036': [1.0, -0.36, 0.1296, -0.220, 0.1564, -0.088, 0.0548, -0.0339],
    'golden_348': [1.0, -0.348, 0.121104, -0.2115, 0.1502, -0.0846, 0.0526, -0.0325],
    'golden_352': [1.0, -0.352, 0.123904, -0.2125, 0.1511, -0.0851, 0.0529, -0.0327],
    'soft_golden_ratio': [1.0, -0.35, 0.1225, -0.214, 0.1519, -0.0855, 0.0532, -0.0329],
    'golden_ratio_balanced': [1.0, -0.35, 0.1225, -0.215, 0.152, -0.0857, 0.0533, -0.033],
    'golden_optimized_1': [1.0, -0.35, 0.1225, -0.2142, 0.1519, -0.0856, 0.0533, -0.033],
    'golden_optimized_2': [1.0, -0.35, 0.1225, -0.2141, 0.1518, -0.0854, 0.0531, -0.0328],
    'golden_optimized_3': [1.0, -0.35, 0.1225, -0.2143, 0.152, -0.0857, 0.0534, -0.0331],
    
    # Oscillatory soft patterns (empirically good for CF)
    'oscillatory_soft': [1.0, -0.35, 0.1225, -0.21, 0.168, -0.084, 0.042, -0.021],
    'oscillatory_soft_v2': [1.0, -0.36, 0.1296, -0.216, 0.1728, -0.0864, 0.0432, -0.0216],
    'oscillatory_soft_v3': [1.0, -0.34, 0.1156, -0.204, 0.1632, -0.0816, 0.0408, -0.0204],
    
    # Fine-tuned coefficients around optimal range
    'soft_tuned_351': [1.0, -0.351, 0.123201, -0.2106, 0.16848, -0.08424, 0.04212, -0.02106],
    'soft_tuned_352': [1.0, -0.352, 0.123904, -0.2112, 0.16896, -0.08448, 0.04224, -0.02112],
    'soft_tuned_353': [1.0, -0.353, 0.124609, -0.2118, 0.16944, -0.08472, 0.04236, -0.02118],
    
    # Natural mathematics patterns
    'fibonacci_soft': [1.0, -0.35, 0.1225, -0.213, 0.1597, -0.0987, 0.061, -0.0377],
    'euler_soft': [1.0, -0.35, 0.1225, -0.2148, 0.1445, -0.0872, 0.0531, -0.0324],
    'natural_harmony': [1.0, -0.35, 0.1225, -0.2142, 0.1528, -0.0866, 0.0541, -0.0334],
    
    # Multi-band for complex patterns
    'multi_band': [1.0, -0.3, 0.2, -0.4, 0.3, -0.2, 0.1, -0.05],
    'multi_band_balanced': [1.0, -0.35, 0.22, -0.38, 0.28, -0.18, 0.09, -0.045],
    
    # Adaptive-like patterns
    'wiener_like': [1.0, -0.35, 0.06, -0.005, 0.0003, -0.00001, 0.0000003, 0.0],
    'adaptive_smooth': [1.0, -0.4, 0.08, -0.008, 0.0006, -0.00003, 0.000001, 0.0],
    
    # Essential baselines
    'identity': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'exponential_decay': [1.0, -0.5, 0.25, -0.125, 0.0625, -0.03125, 0.015625, -0.0078125]
}

# Simplified categories focused on CF
filter_categories = {
    'golden': ['golden_034', 'golden_036', 'golden_348', 'golden_352', 'soft_golden_ratio', 
               'golden_ratio_balanced', 'golden_optimized_1', 'golden_optimized_2', 'golden_optimized_3'],
    'soft': ['oscillatory_soft', 'oscillatory_soft_v2', 'oscillatory_soft_v3', 
             'soft_tuned_351', 'soft_tuned_352', 'soft_tuned_353'],
    'lowpass': ['smooth', 'butterworth', 'gaussian', 'bessel', 'conservative'],
    'mathematical': ['fibonacci_soft', 'euler_soft', 'natural_harmony'],
    'adaptive': ['wiener_like', 'adaptive_smooth', 'multi_band', 'multi_band_balanced'],
    'baseline': ['identity', 'exponential_decay']
}

def get_filter_coefficients(filter_name, order=None, as_tensor=False):
    """Get filter coefficients by name."""
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
    """List available filters by category."""
    if category is None:
        return list(filter_patterns.keys())
    
    if category not in filter_categories:
        raise ValueError(f"Unknown category: {category}. Available: {list(filter_categories.keys())}")
    
    return filter_categories[category]

# =============================================================================
# FILTER DESIGN 1: ORIGINAL UNIVERSAL FILTER
# =============================================================================
class UniversalSpectralFilter(nn.Module):
    def __init__(self, filter_order=6, init_filter_name='smooth'):
        super().__init__()
        self.filter_order = filter_order
        self.init_filter_name = init_filter_name
        
        lowpass = get_filter_coefficients(init_filter_name, as_tensor=True)
        coeffs_data = torch.zeros(filter_order + 1)
        for i, val in enumerate(lowpass[:filter_order + 1]):
            coeffs_data[i] = val

        self.register_buffer('init_coeffs', coeffs_data.clone())
        self.coeffs = nn.Parameter(coeffs_data.clone())
    
    def forward(self, eigenvalues):
        coeffs = self.coeffs
        max_eigenval = torch.max(eigenvalues) + 1e-8
        x = 2 * (eigenvalues / max_eigenval) - 1
        
        result = coeffs[0] * torch.ones_like(x)
        
        if len(coeffs) > 1:
            T_prev, T_curr = torch.ones_like(x), x
            result += coeffs[1] * T_curr
            
            for i in range(2, len(coeffs)):
                T_next = 2 * x * T_curr - T_prev
                result += coeffs[i] * T_next
                T_prev, T_curr = T_curr, T_next
        
        filter_response = torch.exp(-torch.abs(result).clamp(max=10.0)) + 1e-6
        return filter_response

# =============================================================================
# FILTER DESIGN 2: SPECTRAL BASIS FILTER
# =============================================================================
class SpectralBasisFilter(nn.Module):
    """Learnable combination of proven filter patterns"""
    
    def __init__(self, filter_order=6, init_filter_name='smooth'):
        super().__init__()
        self.filter_order = filter_order
        self.init_filter_name = init_filter_name
        
        filter_names = ['golden_036', 'smooth', 'butterworth', 'gaussian', 'bessel', 'conservative']
        
        self.filter_bank = []
        for i, name in enumerate(filter_names):
            coeffs = get_filter_coefficients(name, order=filter_order, as_tensor=True)
            if len(coeffs) < filter_order + 1:
                padded_coeffs = torch.zeros(filter_order + 1)
                padded_coeffs[:len(coeffs)] = coeffs
                coeffs = padded_coeffs
            elif len(coeffs) > filter_order + 1:
                coeffs = coeffs[:filter_order + 1]
            
            self.register_buffer(f'filter_{i}', coeffs)
            self.filter_bank.append(getattr(self, f'filter_{i}'))
        
        init_weights = torch.ones(len(filter_names)) * 0.1
        if init_filter_name in filter_names:
            init_idx = filter_names.index(init_filter_name)
            init_weights[init_idx] = 0.5
        
        self.mixing_weights = nn.Parameter(init_weights)
        self.refinement_coeffs = nn.Parameter(torch.zeros(filter_order + 1))
        self.refinement_scale = nn.Parameter(torch.tensor(0.1))
        self.filter_names = filter_names
    
    def forward(self, eigenvalues):
        weights = torch.softmax(self.mixing_weights, dim=0)
        
        mixed_coeffs = torch.zeros_like(self.filter_bank[0])
        for i, base_filter in enumerate(self.filter_bank):
            mixed_coeffs += weights[i] * base_filter
        
        final_coeffs = mixed_coeffs + self.refinement_scale * self.refinement_coeffs
        
        max_eigenval = torch.max(eigenvalues) + 1e-8
        x = 2 * (eigenvalues / max_eigenval) - 1
        
        result = final_coeffs[0] * torch.ones_like(x)
        if len(final_coeffs) > 1:
            T_prev, T_curr = torch.ones_like(x), x
            result += final_coeffs[1] * T_curr
            
            for i in range(2, len(final_coeffs)):
                T_next = 2 * x * T_curr - T_prev
                result += final_coeffs[i] * T_next
                T_prev, T_curr = T_curr, T_next
        
        return torch.exp(-torch.abs(result).clamp(max=10.0)) + 1e-6
    
    def get_mixing_analysis(self):
        weights = torch.softmax(self.mixing_weights, dim=0).detach().cpu().numpy()
        analysis = {}
        for i, name in enumerate(self.filter_names):
            analysis[name] = weights[i]
        return analysis

# =============================================================================
# FILTER DESIGN 3: ENHANCED SPECTRAL BASIS FILTER
# =============================================================================
class EnhancedSpectralBasisFilter(nn.Module):
    """Enhanced basis filter for maximum performance"""
    
    def __init__(self, filter_order=6, init_filter_name='smooth'):
        super().__init__()
        self.filter_order = filter_order
        self.init_filter_name = init_filter_name
        
        filter_names = [
            'golden_036', 'soft_golden_ratio', 'golden_ratio_balanced', 'golden_optimized_1',
            'smooth', 'butterworth', 'gaussian', 'bessel', 'conservative',
            'fibonacci_soft', 'oscillatory_soft', 'soft_tuned_351', 'soft_tuned_352'
        ]
        
        self.filter_bank = []
        for i, name in enumerate(filter_names):
            try:
                coeffs = get_filter_coefficients(name, order=filter_order, as_tensor=True)
                if len(coeffs) < filter_order + 1:
                    padded_coeffs = torch.zeros(filter_order + 1)
                    padded_coeffs[:len(coeffs)] = coeffs
                    coeffs = padded_coeffs
                elif len(coeffs) > filter_order + 1:
                    coeffs = coeffs[:filter_order + 1]
                
                self.register_buffer(f'filter_{i}', coeffs)
                self.filter_bank.append(getattr(self, f'filter_{i}'))
            except:
                continue
        
        init_weights = torch.ones(len(self.filter_bank)) * 0.02
        
        golden_filters = ['golden_036', 'soft_golden_ratio', 'golden_ratio_balanced', 'golden_optimized_1']
        for i, name in enumerate(filter_names[:len(self.filter_bank)]):
            if name == init_filter_name:
                init_weights[i] = 0.4
            elif name in golden_filters:
                init_weights[i] = 0.15
            elif name in ['smooth', 'butterworth']:
                init_weights[i] = 0.08
        
        init_weights = init_weights / init_weights.sum()
        
        self.mixing_weights = nn.Parameter(init_weights)
        self.refinement_coeffs = nn.Parameter(torch.zeros(filter_order + 1))
        self.refinement_scale = nn.Parameter(torch.tensor(0.2))
        self.transform_scale = nn.Parameter(torch.tensor(1.0))
        self.transform_bias = nn.Parameter(torch.tensor(0.0))
        
        self.filter_names = filter_names[:len(self.filter_bank)]
    
    def forward(self, eigenvalues):
        weights = torch.softmax(self.mixing_weights, dim=0)
        
        mixed_coeffs = torch.zeros_like(self.filter_bank[0])
        for i, base_filter in enumerate(self.filter_bank):
            mixed_coeffs += weights[i] * base_filter
        
        refinement = self.refinement_scale * torch.tanh(self.refinement_coeffs)
        final_coeffs = mixed_coeffs + refinement
        
        max_eigenval = torch.max(eigenvalues) + 1e-8
        x = 2 * (eigenvalues / max_eigenval) - 1
        
        result = final_coeffs[0] * torch.ones_like(x)
        if len(final_coeffs) > 1:
            T_prev, T_curr = torch.ones_like(x), x
            result += final_coeffs[1] * T_curr
            
            for i in range(2, len(final_coeffs)):
                T_next = 2 * x * T_curr - T_prev
                result += final_coeffs[i] * T_next
                T_prev, T_curr = T_curr, T_next
        
        result = self.transform_scale * result + self.transform_bias
        
        return torch.exp(-torch.abs(result).clamp(max=10.0)) + 1e-6
    
    def get_mixing_analysis(self):
        weights = torch.softmax(self.mixing_weights, dim=0).detach().cpu().numpy()
        analysis = {}
        for i, name in enumerate(self.filter_names):
            analysis[name] = weights[i]
        
        sorted_analysis = dict(sorted(analysis.items(), key=lambda x: x[1], reverse=True))
        return sorted_analysis

# =============================================================================
# FILTER DESIGN 4: ADAPTIVE GOLDEN FILTER
# =============================================================================
class AdaptiveGoldenFilter(nn.Module):
    """Learns adaptive variations of golden ratio patterns"""
    
    def __init__(self, filter_order=6, init_filter_name='golden_036'):
        super().__init__()
        self.filter_order = filter_order
        self.init_filter_name = init_filter_name
        
        base_coeffs = get_filter_coefficients('golden_036', as_tensor=True)
        if len(base_coeffs) < filter_order + 1:
            padded_coeffs = torch.zeros(filter_order + 1)
            padded_coeffs[:len(base_coeffs)] = base_coeffs
            base_coeffs = padded_coeffs
        elif len(base_coeffs) > filter_order + 1:
            base_coeffs = base_coeffs[:filter_order + 1]
        
        self.register_buffer('base_coeffs', base_coeffs.clone())
        
        self.scale_factors = nn.Parameter(torch.ones(filter_order + 1))
        self.bias_terms = nn.Parameter(torch.zeros(filter_order + 1) * 0.1)
        self.golden_ratio_delta = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, eigenvalues):
        adaptive_ratio = 0.36 + 0.1 * torch.tanh(self.golden_ratio_delta)
        
        scale_constrained = 0.5 + 0.5 * torch.sigmoid(self.scale_factors)
        bias_constrained = 0.1 * torch.tanh(self.bias_terms)
        
        adapted_coeffs = scale_constrained * self.base_coeffs + bias_constrained
        adapted_coeffs = adapted_coeffs.clone()
        adapted_coeffs[1] = -adaptive_ratio
        
        max_eigenval = torch.max(eigenvalues) + 1e-8
        x = 2 * (eigenvalues / max_eigenval) - 1
        
        result = adapted_coeffs[0] * torch.ones_like(x)
        if len(adapted_coeffs) > 1:
            T_prev, T_curr = torch.ones_like(x), x
            result += adapted_coeffs[1] * T_curr
            
            for i in range(2, len(adapted_coeffs)):
                T_next = 2 * x * T_curr - T_prev
                result += adapted_coeffs[i] * T_next
                T_prev, T_curr = T_curr, T_next
        
        return torch.exp(-torch.abs(result).clamp(max=10.0)) + 1e-6

# =============================================================================
# FILTER DESIGN 5: EIGENVALUE ADAPTIVE FILTER
# =============================================================================
class EigenvalueAdaptiveFilter(nn.Module):
    """Filter that adapts behavior based on eigenvalue magnitude"""
    
    def __init__(self, filter_order=6, init_filter_name='smooth'):
        super().__init__()
        self.filter_order = filter_order
        self.init_filter_name = init_filter_name
        
        base_coeffs = get_filter_coefficients(init_filter_name, as_tensor=True)
        if len(base_coeffs) < 3:
            base_coeffs = torch.cat([base_coeffs, torch.zeros(3 - len(base_coeffs))])
        
        self.low_freq_coeffs = nn.Parameter(base_coeffs[:3].clone())
        self.mid_freq_coeffs = nn.Parameter(base_coeffs[:3].clone())
        self.high_freq_coeffs = nn.Parameter(base_coeffs[:3].clone())
        
        self.boundary_1 = nn.Parameter(torch.tensor(0.3))
        self.boundary_2 = nn.Parameter(torch.tensor(0.7))
        
    def forward(self, eigenvalues):
        max_eigenval = torch.max(eigenvalues) + 1e-8
        norm_eigenvals = eigenvalues / max_eigenval
        
        low_response = self._compute_response(norm_eigenvals, self.low_freq_coeffs)
        mid_response = self._compute_response(norm_eigenvals, self.mid_freq_coeffs)
        high_response = self._compute_response(norm_eigenvals, self.high_freq_coeffs)
        
        boundary_1 = torch.sigmoid(self.boundary_1) * 0.5
        boundary_2 = boundary_1 + torch.sigmoid(self.boundary_2) * 0.5
        
        weight_low = torch.sigmoid((boundary_1 - norm_eigenvals) * 10)
        weight_high = torch.sigmoid((norm_eigenvals - boundary_2) * 10)
        weight_mid = torch.clamp(1 - weight_low - weight_high, min=0.0)
        
        final_response = (weight_low * low_response + 
                         weight_mid * mid_response + 
                         weight_high * high_response)
        
        return torch.clamp(final_response, min=1e-6, max=1.0)
    
    def _compute_response(self, eigenvals, coeffs):
        result = coeffs[0] + coeffs[1] * eigenvals + coeffs[2] * eigenvals**2
        return torch.exp(-torch.abs(result).clamp(max=8.0)) + 1e-6

# =============================================================================
# FILTER DESIGN 6: NEURAL SPECTRAL FILTER
# =============================================================================
class NeuralSpectralFilter(nn.Module):
    """Neural network for spectral response learning"""
    
    def __init__(self, filter_order=6, init_filter_name='smooth'):
        super().__init__()
        self.filter_order = filter_order
        self.init_filter_name = init_filter_name
        
        self.filter_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16), 
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        with torch.no_grad():
            self.filter_net[-2].weight.normal_(0, 0.1)
            self.filter_net[-2].bias.fill_(-1.0)
        
    def forward(self, eigenvalues):
        max_eigenval = torch.max(eigenvalues) + 1e-8
        norm_eigenvals = (eigenvalues / max_eigenval).unsqueeze(-1)
        
        filter_response = self.filter_net(norm_eigenvals).squeeze(-1)
        
        return filter_response + 1e-6

# =============================================================================
# NEW: HIGH-CAPACITY FILTER DESIGNS
# =============================================================================

class DeepSpectralFilter(nn.Module):
    """Deep neural network for spectral response learning (~1000+ parameters)"""
    
    def __init__(self, filter_order=6, init_filter_name='smooth', hidden_dims=[64, 32, 16]):
        super().__init__()
        self.filter_order = filter_order
        self.init_filter_name = init_filter_name
        
        layers = []
        prev_dim = 1
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.filter_net = nn.Sequential(*layers)
        self._initialize_network()
    
    def _initialize_network(self):
        for module in self.filter_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.1)
                nn.init.constant_(module.bias, 0.0)
        
        final_layer = self.filter_net[-2]
        nn.init.constant_(final_layer.bias, -1.0)
    
    def forward(self, eigenvalues):
        max_eigenval = torch.max(eigenvalues) + 1e-8
        norm_eigenvals = (eigenvalues / max_eigenval).unsqueeze(-1)
        
        filter_response = self.filter_net(norm_eigenvals).squeeze(-1)
        return filter_response + 1e-6

class MultiScaleSpectralFilter(nn.Module):
    """Multi-scale spectral filtering with learnable frequency bands (~500+ parameters)"""
    
    def __init__(self, filter_order=6, init_filter_name='smooth', n_bands=8):
        super().__init__()
        self.filter_order = filter_order
        self.init_filter_name = init_filter_name
        self.n_bands = n_bands
        
        init_boundaries = torch.linspace(0, 1, n_bands + 1)[1:-1]
        self.band_boundaries = nn.Parameter(init_boundaries)
        self.band_responses = nn.Parameter(torch.ones(n_bands) * 0.5)
        self.transition_sharpness = nn.Parameter(torch.tensor(10.0))
        
        self.modulation_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()
        )
        
        self.modulation_strength = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, eigenvalues):
        max_eigenval = torch.max(eigenvalues) + 1e-8
        norm_eigenvals = eigenvalues / max_eigenval
        
        sorted_boundaries = torch.sort(self.band_boundaries)[0]
        boundaries = torch.cat([torch.zeros(1, device=eigenvalues.device), 
                               sorted_boundaries,
                               torch.ones(1, device=eigenvalues.device)])
        
        sharpness = torch.abs(self.transition_sharpness) + 1.0
        band_responses = torch.sigmoid(self.band_responses)
        
        response = torch.zeros_like(norm_eigenvals)
        
        for i in range(self.n_bands):
            left_boundary = boundaries[i]
            right_boundary = boundaries[i + 1]
            
            left_transition = torch.sigmoid(sharpness * (norm_eigenvals - left_boundary))
            right_transition = torch.sigmoid(sharpness * (right_boundary - norm_eigenvals))
            
            band_membership = left_transition * right_transition
            response += band_membership * band_responses[i]
        
        modulation_input = norm_eigenvals.unsqueeze(-1)
        modulation = self.modulation_net(modulation_input).squeeze(-1)
        modulation_scale = torch.sigmoid(self.modulation_strength)
        
        final_response = response + modulation_scale * modulation
        
        return torch.clamp(final_response, min=1e-6, max=1.0)

class EnsembleSpectralFilter(nn.Module):
    """Ensemble of different filter types with learned mixing (~2000+ parameters)"""
    
    def __init__(self, filter_order=6, init_filter_name='smooth'):
        super().__init__()
        self.filter_order = filter_order
        self.init_filter_name = init_filter_name
        
        self.classical_filter = UniversalSpectralFilter(filter_order, init_filter_name)
        self.deep_filter = DeepSpectralFilter(filter_order, init_filter_name, [32, 16])
        self.multiscale_filter = MultiScaleSpectralFilter(filter_order, init_filter_name, 6)
        
        self.ensemble_logits = nn.Parameter(torch.ones(3))
        self.ensemble_temperature = nn.Parameter(torch.tensor(1.0))
        
        self.meta_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Tanh()
        )
    
    def forward(self, eigenvalues):
        classical_response = self.classical_filter(eigenvalues)
        deep_response = self.deep_filter(eigenvalues)
        multiscale_response = self.multiscale_filter(eigenvalues)
        
        eigenval_stats = torch.stack([
            eigenvalues.mean(),
            eigenvalues.std(),
            eigenvalues.max()
        ])
        
        meta_adjustments = self.meta_net(eigenval_stats)
        adjusted_logits = self.ensemble_logits + 0.5 * meta_adjustments
        
        temperature = torch.abs(self.ensemble_temperature) + 0.1
        ensemble_weights = torch.softmax(adjusted_logits / temperature, dim=0)
        
        final_response = (ensemble_weights[0] * classical_response +
                         ensemble_weights[1] * deep_response +
                         ensemble_weights[2] * multiscale_response)
        
        return final_response
    
    def get_ensemble_analysis(self):
        with torch.no_grad():
            weights = torch.softmax(self.ensemble_logits, dim=0)
            return {
                'classical': weights[0].item(),
                'deep': weights[1].item(), 
                'multiscale': weights[2].item(),
                'temperature': torch.abs(self.ensemble_temperature).item()
            }