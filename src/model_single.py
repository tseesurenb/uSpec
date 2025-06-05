'''
Created on June 3, 2025
PyTorch Implementation of uSpec: Universal Spectral Collaborative Filtering
MULTIPLE FILTER DESIGNS: Original, Basis, Adaptive, Neural

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import time
import gc
import filters as fl

# =============================================================================
# FILTER DESIGN 1: ORIGINAL UNIVERSAL FILTER (Enhanced)
# =============================================================================
class UniversalSpectralFilter(nn.Module):
    def __init__(self, filter_order=6, init_filter_name='smooth'):
        super().__init__()
        self.filter_order = filter_order
        self.init_filter_name = init_filter_name
        
        # Initialize with specified filter from filters.py
        lowpass = fl.get_filter_coefficients(init_filter_name, as_tensor=True)
        coeffs_data = torch.zeros(filter_order + 1)
        for i, val in enumerate(lowpass[:filter_order + 1]):
            coeffs_data[i] = val

        print(f"Initializing UniversalSpectralFilter with order {filter_order}")
        print(f"  Initial filter: {init_filter_name}")
        print(f"  Initial coefficients: {coeffs_data.cpu().numpy()}")
        
        # Store initial coefficients for comparison
        self.register_buffer('init_coeffs', coeffs_data.clone())
        
        # Direct learning with proper initialization
        self.coeffs = nn.Parameter(coeffs_data.clone())
        
        print(f"  Mode: Enhanced direct coefficient learning")
    
    def forward(self, eigenvalues):
        """Apply learnable spectral filter using Chebyshev polynomials"""
        coeffs = self.coeffs
        
        # Normalize eigenvalues to [-1, 1]
        max_eigenval = torch.max(eigenvalues) + 1e-8
        x = 2 * (eigenvalues / max_eigenval) - 1
        
        # Compute Chebyshev polynomial response
        result = coeffs[0] * torch.ones_like(x)
        
        if len(coeffs) > 1:
            T_prev, T_curr = torch.ones_like(x), x
            result += coeffs[1] * T_curr
            
            for i in range(2, len(coeffs)):
                T_next = 2 * x * T_curr - T_prev
                result += coeffs[i] * T_next
                T_prev, T_curr = T_curr, T_next
        
        # Exponential activation for spectral filtering
        filter_response = torch.exp(-torch.abs(result).clamp(max=10.0)) + 1e-6
        
        return filter_response

# =============================================================================
# FILTER DESIGN 2: SPECTRAL BASIS FILTER (Recommended)
# =============================================================================
class SpectralBasisFilter(nn.Module):
    """Use learnable combination of known good filter patterns"""
    
    def __init__(self, filter_order=6, init_filter_name='smooth'):
        super().__init__()
        self.filter_order = filter_order
        self.init_filter_name = init_filter_name
        
        # Pre-define several good filter patterns from filters.py
        filter_names = ['golden_036', 'smooth', 'butterworth', 'gaussian', 'bessel', 'conservative']
        
        self.filter_bank = []
        for i, name in enumerate(filter_names):
            coeffs = fl.get_filter_coefficients(name, order=filter_order, as_tensor=True)
            # Pad or truncate to exact size
            if len(coeffs) < filter_order + 1:
                padded_coeffs = torch.zeros(filter_order + 1)
                padded_coeffs[:len(coeffs)] = coeffs
                coeffs = padded_coeffs
            elif len(coeffs) > filter_order + 1:
                coeffs = coeffs[:filter_order + 1]
            
            self.register_buffer(f'filter_{i}', coeffs)
            self.filter_bank.append(getattr(self, f'filter_{i}'))
        
        # Learnable mixing weights - initialize based on init_filter_name
        init_weights = torch.ones(len(filter_names)) * 0.1
        if init_filter_name in filter_names:
            init_idx = filter_names.index(init_filter_name)
            init_weights[init_idx] = 0.5
        
        self.mixing_weights = nn.Parameter(init_weights)
        
        # Learnable refinement on top of the mixture
        self.refinement_coeffs = nn.Parameter(torch.zeros(filter_order + 1))
        self.refinement_scale = nn.Parameter(torch.tensor(0.1))  # Small refinements
        
        # Store for debugging
        self.filter_names = filter_names
        
        print(f"Initializing SpectralBasisFilter with {len(filter_names)} base filters")
        print(f"  Base filters: {filter_names}")
        print(f"  Primary initialization: {init_filter_name}")
        print(f"  Mode: Learnable basis combination + refinement")
    
    def forward(self, eigenvalues):
        # Mix the pre-defined filters
        weights = torch.softmax(self.mixing_weights, dim=0)
        
        mixed_coeffs = torch.zeros_like(self.filter_bank[0])
        for i, base_filter in enumerate(self.filter_bank):
            mixed_coeffs += weights[i] * base_filter
        
        # Add learnable refinement
        final_coeffs = mixed_coeffs + self.refinement_scale * self.refinement_coeffs
        
        # Apply as Chebyshev polynomial (like original)
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
        """Return analysis of learned mixing weights"""
        weights = torch.softmax(self.mixing_weights, dim=0).detach().cpu().numpy()
        analysis = {}
        for i, name in enumerate(self.filter_names):
            analysis[name] = weights[i]
        return analysis

# =============================================================================
# FILTER DESIGN 3: EIGENVALUE ADAPTIVE FILTER
# =============================================================================
class EigenvalueAdaptiveFilter(nn.Module):
    """Filter that adapts behavior based on eigenvalue magnitude"""
    
    def __init__(self, filter_order=6, init_filter_name='smooth'):
        super().__init__()
        self.filter_order = filter_order
        self.init_filter_name = init_filter_name
        
        # Initialize from a base filter and derive adaptive parameters
        base_coeffs = fl.get_filter_coefficients(init_filter_name, as_tensor=True)
        if len(base_coeffs) < 3:
            base_coeffs = torch.cat([base_coeffs, torch.zeros(3 - len(base_coeffs))])
        
        # Parameters for different eigenvalue ranges
        self.low_freq_coeffs = nn.Parameter(base_coeffs[:3].clone())    # For λ ∈ [0, boundary_1]
        self.mid_freq_coeffs = nn.Parameter(base_coeffs[:3].clone())    # For λ ∈ [boundary_1, boundary_2]  
        self.high_freq_coeffs = nn.Parameter(base_coeffs[:3].clone())   # For λ ∈ [boundary_2, 1.0]
        
        # Learnable transition boundaries
        self.boundary_1 = nn.Parameter(torch.tensor(0.3))
        self.boundary_2 = nn.Parameter(torch.tensor(0.7))
        
        print(f"Initializing EigenvalueAdaptiveFilter")
        print(f"  Base filter: {init_filter_name}")
        print(f"  Mode: Eigenvalue-adaptive filtering")
        
    def forward(self, eigenvalues):
        # Normalize eigenvalues
        max_eigenval = torch.max(eigenvalues) + 1e-8
        norm_eigenvals = eigenvalues / max_eigenval
        
        # Compute responses for each frequency band
        low_response = self._compute_response(norm_eigenvals, self.low_freq_coeffs)
        mid_response = self._compute_response(norm_eigenvals, self.mid_freq_coeffs)
        high_response = self._compute_response(norm_eigenvals, self.high_freq_coeffs)
        
        # Constrain boundaries to reasonable ranges
        boundary_1 = torch.sigmoid(self.boundary_1) * 0.5  # [0, 0.5]
        boundary_2 = boundary_1 + torch.sigmoid(self.boundary_2) * 0.5  # [boundary_1, boundary_1+0.5]
        
        # Smooth interpolation between frequency bands
        weight_low = torch.sigmoid((boundary_1 - norm_eigenvals) * 10)
        weight_high = torch.sigmoid((norm_eigenvals - boundary_2) * 10)
        weight_mid = torch.clamp(1 - weight_low - weight_high, min=0.0)
        
        final_response = (weight_low * low_response + 
                         weight_mid * mid_response + 
                         weight_high * high_response)
        
        return torch.clamp(final_response, min=1e-6, max=1.0)
    
    def _compute_response(self, eigenvals, coeffs):
        """Simple polynomial response"""
        result = coeffs[0] + coeffs[1] * eigenvals + coeffs[2] * eigenvals**2
        return torch.exp(-torch.abs(result).clamp(max=8.0)) + 1e-6

# =============================================================================
# FILTER DESIGN 4: NEURAL SPECTRAL FILTER
# =============================================================================
class NeuralSpectralFilter(nn.Module):
    """Use a small neural network to learn the spectral response directly"""
    
    def __init__(self, filter_order=6, init_filter_name='smooth'):
        super().__init__()
        self.filter_order = filter_order
        self.init_filter_name = init_filter_name
        
        # Small MLP to map eigenvalue -> filter response
        self.filter_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16), 
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        # Initialize to roughly match exponential decay behavior
        with torch.no_grad():
            # Initialize to produce reasonable spectral filter responses
            self.filter_net[-2].weight.normal_(0, 0.1)
            self.filter_net[-2].bias.fill_(-1.0)  # Bias toward lower values
        
        print(f"Initializing NeuralSpectralFilter")
        print(f"  Base filter inspiration: {init_filter_name}")
        print(f"  Mode: Neural network spectral response")
        
    def forward(self, eigenvalues):
        # Normalize eigenvalues to [0, 1]
        max_eigenval = torch.max(eigenvalues) + 1e-8
        norm_eigenvals = (eigenvalues / max_eigenval).unsqueeze(-1)
        
        # Pass through neural network
        filter_response = self.filter_net(norm_eigenvals).squeeze(-1)
        
        return filter_response + 1e-6

# =============================================================================
# MAIN MODEL CLASS WITH FILTER SELECTION
# =============================================================================
class UniversalSpectralCF(nn.Module):
    def __init__(self, adj_mat, config=None):
        super().__init__()
        
        # Configuration
        self.config = config or {}
        self.device = self.config.get('device', 'cpu')
        self.n_eigen = self.config.get('n_eigen', 50)
        self.filter_order = self.config.get('filter_order', 6)
        self.filter = self.config.get('filter', 'ui')
        
        # NEW: Filter design selection
        self.filter_design = self.config.get('filter_design', 'basis')  # 'original', 'basis', 'adaptive', 'neural'
        self.init_filter = self.config.get('init_filter', 'smooth')
        
        print(f"Filter Design: {self.filter_design}")
        print(f"Initialization: {self.init_filter}")
        
        # Convert and register adjacency matrix
        adj_dense = adj_mat.toarray() if sp.issparse(adj_mat) else adj_mat
        self.register_buffer('adj_tensor', torch.tensor(adj_dense, dtype=torch.float32))
        self.n_users, self.n_items = self.adj_tensor.shape
        
        # Compute and register normalized adjacency
        row_sums = self.adj_tensor.sum(dim=1, keepdim=True) + 1e-8
        col_sums = self.adj_tensor.sum(dim=0, keepdim=True) + 1e-8
        norm_adj = self.adj_tensor / torch.sqrt(row_sums) / torch.sqrt(col_sums)
        self.register_buffer('norm_adj', norm_adj)
        
        # Clean up intermediate variables
        del adj_dense, row_sums, col_sums
        self._memory_cleanup()
        
        # Initialize filters and weights
        self._setup_filters()
        self._setup_combination_weights()
    
    def _memory_cleanup(self):
        """Force memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _setup_filters(self):
        """Setup spectral filters with selected design"""
        print(f"Computing eigendecompositions for filter type: {self.filter}")
        start = time.time()
        
        self.user_filter = None
        self.item_filter = None
        
        if self.filter in ['u', 'ui']:
            print("Processing user similarity...")
            self.user_filter = self._create_filter_memory_efficient('user')
            self._memory_cleanup()
        
        if self.filter in ['i', 'ui']:
            print("Processing item similarity...")
            self.item_filter = self._create_filter_memory_efficient('item')
            self._memory_cleanup()
        
        print(f'Filter setup completed in {time.time() - start:.2f}s')
    
    def _create_filter_memory_efficient(self, filter_type):
        """Create filter with memory-efficient eigendecomposition"""
        print(f"  Computing {filter_type} similarity matrix...")
        
        with torch.no_grad():
            if filter_type == 'user':
                similarity_matrix = self._compute_similarity_chunked(
                    self.norm_adj, self.norm_adj.t(), chunk_size=1000
                )
                n_components = self.n_users
            else:
                similarity_matrix = self._compute_similarity_chunked(
                    self.norm_adj.t(), self.norm_adj, chunk_size=1000
                )
                n_components = self.n_items
        
        print(f"  Converting to numpy and computing eigendecomposition...")
        sim_np = similarity_matrix.cpu().numpy()
        
        del similarity_matrix
        self._memory_cleanup()
        
        k = min(self.n_eigen, n_components - 2)
        
        try:
            print(f"  Computing {k} eigenvalues...")
            eigenvals, eigenvecs = eigsh(sp.csr_matrix(sim_np), k=k, which='LM')
            
            self.register_buffer(f'{filter_type}_eigenvals', 
                               torch.tensor(np.real(eigenvals), dtype=torch.float32))
            self.register_buffer(f'{filter_type}_eigenvecs', 
                               torch.tensor(np.real(eigenvecs), dtype=torch.float32))
            
            print(f"  {filter_type.capitalize()} eigendecomposition: {k} components")
            
        except Exception as e:
            print(f"  {filter_type.capitalize()} eigendecomposition failed: {e}")
            print(f"  Using fallback identity matrices...")
            
            self.register_buffer(f'{filter_type}_eigenvals', 
                               torch.ones(min(self.n_eigen, n_components)))
            self.register_buffer(f'{filter_type}_eigenvecs', 
                               torch.eye(n_components, min(self.n_eigen, n_components)))
        
        del sim_np
        if 'eigenvals' in locals():
            del eigenvals, eigenvecs
        self._memory_cleanup()
        
        # Update this section in your _create_filter_memory_efficient method:

        # CREATE FILTER BASED ON SELECTED DESIGN
        if self.filter_design == 'original':
            return UniversalSpectralFilter(self.filter_order, self.init_filter)
        elif self.filter_design == 'basis':
            return SpectralBasisFilter(self.filter_order, self.init_filter)
        elif self.filter_design == 'enhanced_basis':
            return EnhancedSpectralBasisFilter(self.filter_order, self.init_filter)
        elif self.filter_design == 'adaptive_golden':
            return AdaptiveGoldenFilter(self.filter_order, self.init_filter)
        elif self.filter_design == 'adaptive':
            return EigenvalueAdaptiveFilter(self.filter_order, self.init_filter)
        elif self.filter_design == 'neural':
            return NeuralSpectralFilter(self.filter_order, self.init_filter)
        else:
            raise ValueError(f"Unknown filter design: {self.filter_design}")
            
    def _compute_similarity_chunked(self, A, B, chunk_size=1000):
        """Compute A @ B in chunks to save memory"""
        if A.shape[0] <= chunk_size:
            return A @ B
        
        print(f"    Using chunked computation (chunk_size={chunk_size})...")
        result_chunks = []
        
        for i in range(0, A.shape[0], chunk_size):
            end_idx = min(i + chunk_size, A.shape[0])
            chunk_result = A[i:end_idx] @ B
            result_chunks.append(chunk_result)
            
            if i > 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return torch.cat(result_chunks, dim=0)
    
    def _setup_combination_weights(self):
        """Setup learnable combination weights"""
        init_weights = {
            'u': [0.5, 0.5],
            'i': [0.5, 0.5], 
            'ui': [0.5, 0.3, 0.2]
        }
        self.combination_weights = nn.Parameter(torch.tensor(init_weights[self.filter]))
    
    def _get_filter_matrices(self):
        """Compute spectral filter matrices with memory management"""
        user_matrix = item_matrix = None
        
        if self.user_filter is not None:
            response = self.user_filter(self.user_eigenvals)
            user_matrix = self.user_eigenvecs @ torch.diag(response) @ self.user_eigenvecs.t()
        
        if self.item_filter is not None:
            response = self.item_filter(self.item_eigenvals)
            item_matrix = self.item_eigenvecs @ torch.diag(response) @ self.item_eigenvecs.t()
        
        return user_matrix, item_matrix
    
    def forward(self, users):
        """Clean forward pass - ONLY returns predictions"""
        user_profiles = self.adj_tensor[users]
        
        user_filter_matrix, item_filter_matrix = self._get_filter_matrices()
        
        scores = [user_profiles]
        
        if self.filter in ['i', 'ui'] and item_filter_matrix is not None:
            scores.append(user_profiles @ item_filter_matrix)
        
        if self.filter in ['u', 'ui'] and user_filter_matrix is not None:
            user_filtered = user_filter_matrix[users] @ self.adj_tensor
            scores.append(user_filtered)
        
        weights = torch.softmax(self.combination_weights, dim=0)
        predicted = sum(w * score for w, score in zip(weights, scores))
        
        if self.training and (self.n_users > 10000 or self.n_items > 10000):
            del user_filter_matrix, item_filter_matrix
            self._memory_cleanup()
        
        return predicted
    
    def getUsersRating(self, batch_users):
        """Memory-efficient evaluation interface"""
        self.eval()
        with torch.no_grad():
            if isinstance(batch_users, np.ndarray):
                batch_users = torch.LongTensor(batch_users)
            
            result = self.forward(batch_users).cpu().numpy()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return result
    
    def get_filter_parameters(self):
        """Get filter parameters for separate optimization"""
        filter_params = []
        if self.user_filter is not None:
            filter_params.extend(self.user_filter.parameters())
        if self.item_filter is not None:
            filter_params.extend(self.item_filter.parameters())
        return filter_params
    
    def get_other_parameters(self):
        """Get non-filter parameters"""
        filter_param_ids = {id(p) for p in self.get_filter_parameters()}
        return [p for p in self.parameters() if id(p) not in filter_param_ids]
    
    def debug_filter_learning(self):
        """Enhanced debug for different filter designs"""
        print(f"\n=== FILTER LEARNING DEBUG ({self.filter_design.upper()}) ===")
        
        with torch.no_grad():
            if self.filter in ['u', 'ui'] and self.user_filter is not None:
                print(f"\n👤 User Filter ({self.filter_design}):")
                self._debug_single_filter(self.user_filter, "User")
            
            if self.filter in ['i', 'ui'] and self.item_filter is not None:
                print(f"\n🎬 Item Filter ({self.filter_design}):")
                self._debug_single_filter(self.item_filter, "Item")
            
            # Combination weights
            weights = torch.softmax(self.combination_weights, dim=0)
            print(f"\n🔗 Combination Weights: {weights.cpu().numpy()}")
        
        print("=== END DEBUG ===\n")
    
    def _debug_single_filter(self, filter_obj, filter_name):
        """Debug individual filter based on its type"""
        if isinstance(filter_obj, UniversalSpectralFilter):
            init_coeffs = filter_obj.init_coeffs.cpu().numpy()
            current_coeffs = filter_obj.coeffs.cpu().numpy()
            
            print(f"  Initial filter: {filter_obj.init_filter_name}")
            print(f"  Initial coeffs: {init_coeffs}")
            print(f"  Current coeffs: {current_coeffs}")
            
            change = current_coeffs - init_coeffs
            abs_change = np.abs(change)
            print(f"  Absolute Δ:     {change}")
            print(f"  Max |Δ|:        {abs_change.max():.6f}")
            print(f"  Mean |Δ|:       {abs_change.mean():.6f}")
            
        elif isinstance(filter_obj, SpectralBasisFilter):
            mixing_analysis = filter_obj.get_mixing_analysis()
            print(f"  Base filter mixing weights:")
            for name, weight in mixing_analysis.items():
                print(f"    {name:12}: {weight:.4f}")
            
            refinement_scale = filter_obj.refinement_scale.cpu().item()
            refinement_coeffs = filter_obj.refinement_coeffs.cpu().numpy()
            print(f"  Refinement scale: {refinement_scale:.4f}")
            print(f"  Refinement coeffs: {refinement_coeffs}")
            
        elif isinstance(filter_obj, EigenvalueAdaptiveFilter):
            boundary_1 = torch.sigmoid(filter_obj.boundary_1).cpu().item() * 0.5
            boundary_2 = boundary_1 + torch.sigmoid(filter_obj.boundary_2).cpu().item() * 0.5
            print(f"  Learned boundaries: {boundary_1:.3f}, {boundary_2:.3f}")
            print(f"  Low freq coeffs:  {filter_obj.low_freq_coeffs.cpu().numpy()}")
            print(f"  Mid freq coeffs:  {filter_obj.mid_freq_coeffs.cpu().numpy()}")
            print(f"  High freq coeffs: {filter_obj.high_freq_coeffs.cpu().numpy()}")
            
        elif isinstance(filter_obj, NeuralSpectralFilter):
            print(f"  Neural filter - {sum(p.numel() for p in filter_obj.parameters())} parameters")
            print(f"  Network structure: {filter_obj.filter_net}")

    def get_memory_usage(self):
        """Get current memory usage statistics"""
        stats = {
            'parameters': sum(p.numel() * p.element_size() for p in self.parameters()) / 1024**2,
            'buffers': sum(b.numel() * b.element_size() for b in self.buffers()) / 1024**2,
        }
        
        if torch.cuda.is_available():
            stats['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**2
            stats['gpu_cached'] = torch.cuda.memory_reserved() / 1024**2
        
        return {k: f"{v:.2f} MB" for k, v in stats.items()}
    
    def cleanup_memory(self):
        """Manual memory cleanup method"""
        self._memory_cleanup()
        print(f"Memory cleaned. Current usage: {self.get_memory_usage()}")


# Enhanced Spectral Basis Filter - Optimized for Maximum Performance

class EnhancedSpectralBasisFilter(nn.Module):
    """Enhanced basis filter that can achieve peak performance while maintaining consistency"""
    
    def __init__(self, filter_order=6, init_filter_name='smooth'):
        super().__init__()
        self.filter_order = filter_order
        self.init_filter_name = init_filter_name
        
        # Expanded filter bank with more proven high-performance filters
        filter_names = [
            'golden_036', 'soft_golden_ratio', 'golden_ratio_soft_v2', 'golden_ratio_balanced',
            'smooth', 'butterworth', 'gaussian', 'bessel', 'conservative',
            'fibonacci_soft', 'oscillatory_soft', 'soft_tuned_351', 'soft_tuned_352'
        ]
        
        self.filter_bank = []
        for i, name in enumerate(filter_names):
            try:
                coeffs = fl.get_filter_coefficients(name, order=filter_order, as_tensor=True)
                # Pad or truncate to exact size
                if len(coeffs) < filter_order + 1:
                    padded_coeffs = torch.zeros(filter_order + 1)
                    padded_coeffs[:len(coeffs)] = coeffs
                    coeffs = padded_coeffs
                elif len(coeffs) > filter_order + 1:
                    coeffs = coeffs[:filter_order + 1]
                
                self.register_buffer(f'filter_{i}', coeffs)
                self.filter_bank.append(getattr(self, f'filter_{i}'))
            except:
                # Skip filters that don't exist
                continue
        
        # ENHANCED: Smarter initialization that favors high-performance filters
        init_weights = torch.ones(len(self.filter_bank)) * 0.02  # Very low base weight
        
        # Give much higher initial weights to known good filters
        golden_filters = ['golden_036', 'soft_golden_ratio', 'golden_ratio_soft_v2', 'golden_ratio_balanced']
        for i, name in enumerate(filter_names[:len(self.filter_bank)]):
            if name == init_filter_name:
                init_weights[i] = 0.4  # Primary initialization gets high weight
            elif name in golden_filters:
                init_weights[i] = 0.15  # Golden variants get higher weight
            elif name in ['smooth', 'butterworth']:
                init_weights[i] = 0.08  # Known good filters get medium weight
        
        # Normalize to sum to 1
        init_weights = init_weights / init_weights.sum()
        
        self.mixing_weights = nn.Parameter(init_weights)
        
        # ENHANCED: More powerful refinement
        self.refinement_coeffs = nn.Parameter(torch.zeros(filter_order + 1))
        self.refinement_scale = nn.Parameter(torch.tensor(0.2))  # Allow larger refinements
        
        # ENHANCED: Add learnable non-linear transformation
        self.transform_scale = nn.Parameter(torch.tensor(1.0))
        self.transform_bias = nn.Parameter(torch.tensor(0.0))
        
        # Store for debugging
        self.filter_names = filter_names[:len(self.filter_bank)]
        
        print(f"Initializing EnhancedSpectralBasisFilter with {len(self.filter_bank)} base filters")
        print(f"  Base filters: {self.filter_names}")
        print(f"  Primary initialization: {init_filter_name}")
        print(f"  Initial weights for golden filters: {[init_weights[i].item() for i, name in enumerate(self.filter_names) if 'golden' in name]}")
        print(f"  Mode: Enhanced learnable basis combination")
    
    def forward(self, eigenvalues):
        # ENHANCED: Use softmax with temperature for more flexible mixing
        temperature = 1.0  # Could make this learnable too
        weights = torch.softmax(self.mixing_weights / temperature, dim=0)
        
        # Mix the pre-defined filters
        mixed_coeffs = torch.zeros_like(self.filter_bank[0])
        for i, base_filter in enumerate(self.filter_bank):
            mixed_coeffs += weights[i] * base_filter
        
        # ENHANCED: More sophisticated refinement
        refinement = self.refinement_scale * torch.tanh(self.refinement_coeffs)  # Constrain refinement
        final_coeffs = mixed_coeffs + refinement
        
        # Apply as Chebyshev polynomial
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
        
        # ENHANCED: Learnable transformation for better expressiveness
        result = self.transform_scale * result + self.transform_bias
        
        return torch.exp(-torch.abs(result).clamp(max=10.0)) + 1e-6
    
    def get_mixing_analysis(self):
        """Return detailed analysis of learned mixing weights"""
        weights = torch.softmax(self.mixing_weights, dim=0).detach().cpu().numpy()
        analysis = {}
        for i, name in enumerate(self.filter_names):
            analysis[name] = weights[i]
        
        # Sort by weight for easy interpretation
        sorted_analysis = dict(sorted(analysis.items(), key=lambda x: x[1], reverse=True))
        return sorted_analysis


# ALTERNATIVE: Adaptive Golden Filter (learns variations of golden ratio patterns)
class AdaptiveGoldenFilter(nn.Module):
    """Learns adaptive variations of golden ratio patterns for maximum performance"""
    
    def __init__(self, filter_order=6, init_filter_name='golden_036'):
        super().__init__()
        self.filter_order = filter_order
        self.init_filter_name = init_filter_name
        
        # Start with golden_036 as base and learn variations
        base_coeffs = fl.get_filter_coefficients('golden_036', as_tensor=True)
        if len(base_coeffs) < filter_order + 1:
            padded_coeffs = torch.zeros(filter_order + 1)
            padded_coeffs[:len(base_coeffs)] = base_coeffs
            base_coeffs = padded_coeffs
        elif len(base_coeffs) > filter_order + 1:
            base_coeffs = base_coeffs[:filter_order + 1]
        
        # Store base as non-trainable
        self.register_buffer('base_coeffs', base_coeffs.clone())
        
        # LEARNABLE: Multiplicative and additive adaptations
        self.scale_factors = nn.Parameter(torch.ones(filter_order + 1))
        self.bias_terms = nn.Parameter(torch.zeros(filter_order + 1) * 0.1)
        
        # LEARNABLE: Golden ratio variation
        self.golden_ratio_delta = nn.Parameter(torch.tensor(0.0))  # Learn deviation from 0.36
        
        print(f"Initializing AdaptiveGoldenFilter")
        print(f"  Base: golden_036 coefficients")
        print(f"  Mode: Learnable golden ratio variations")
    
    def forward(self, eigenvalues):
        # Compute adaptive golden ratio
        adaptive_ratio = 0.36 + 0.1 * torch.tanh(self.golden_ratio_delta)  # 0.26 to 0.46 range
        
        # Scale base coefficients adaptively
        scale_constrained = 0.5 + 0.5 * torch.sigmoid(self.scale_factors)  # 0.5 to 1.0 range
        bias_constrained = 0.1 * torch.tanh(self.bias_terms)  # -0.1 to 0.1 range
        
        # Compute final coefficients
        adapted_coeffs = scale_constrained * self.base_coeffs + bias_constrained
        
        # Override second coefficient with learned golden ratio
        adapted_coeffs = adapted_coeffs.clone()
        adapted_coeffs[1] = -adaptive_ratio
        
        # Apply as Chebyshev polynomial.
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