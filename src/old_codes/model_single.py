'''
Created on June 3, 2025
PyTorch Implementation of uSpec: Universal Spectral Collaborative Filtering
MULTIPLE FILTER DESIGNS: Original, Basis, Adaptive, Neural, Deep, MultiScale, Ensemble

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


class UniversalSpectralCF(nn.Module):
    def __init__(self, adj_mat, config=None):
        super().__init__()
        
        # Configuration
        self.config = config or {}
        self.device = self.config.get('device', 'cpu')
        self.n_eigen = self.config.get('n_eigen', 50)
        self.filter_order = self.config.get('filter_order', 6)
        self.filter = self.config.get('filter', 'ui')
        
        # Filter design selection with new high-capacity options
        self.filter_design = self.config.get('filter_design', 'basis')
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
        
        # CREATE FILTER BASED ON SELECTED DESIGN (UPDATED with new filters)
        if self.filter_design == 'original':
            return fl.UniversalSpectralFilter(self.filter_order, self.init_filter)
        elif self.filter_design == 'basis':
            return fl.SpectralBasisFilter(self.filter_order, self.init_filter)
        elif self.filter_design == 'enhanced_basis':
            return fl.EnhancedSpectralBasisFilter(self.filter_order, self.init_filter)
        elif self.filter_design == 'adaptive_golden':
            return fl.AdaptiveGoldenFilter(self.filter_order, self.init_filter)
        elif self.filter_design == 'adaptive':
            return fl.EigenvalueAdaptiveFilter(self.filter_order, self.init_filter)
        elif self.filter_design == 'neural':
            return fl.NeuralSpectralFilter(self.filter_order, self.init_filter)
        # NEW HIGH-CAPACITY FILTERS
        elif self.filter_design == 'deep':
            return fl.DeepSpectralFilter(self.filter_order, self.init_filter)
        elif self.filter_design == 'multiscale':
            return fl.MultiScaleSpectralFilter(self.filter_order, self.init_filter)
        elif self.filter_design == 'ensemble':
            return fl.EnsembleSpectralFilter(self.filter_order, self.init_filter)
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
        """Debug individual filter based on its type (UPDATED with new filters)"""
        if isinstance(filter_obj, fl.UniversalSpectralFilter):
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
            
        elif isinstance(filter_obj, fl.SpectralBasisFilter):
            mixing_analysis = filter_obj.get_mixing_analysis()
            print(f"  Base filter mixing weights:")
            for name, weight in mixing_analysis.items():
                print(f"    {name:12}: {weight:.4f}")
            
            refinement_scale = filter_obj.refinement_scale.cpu().item()
            refinement_coeffs = filter_obj.refinement_coeffs.cpu().numpy()
            print(f"  Refinement scale: {refinement_scale:.4f}")
            print(f"  Refinement coeffs: {refinement_coeffs}")
            
        elif isinstance(filter_obj, fl.EnhancedSpectralBasisFilter):
            mixing_analysis = filter_obj.get_mixing_analysis()
            print(f"  Enhanced base filter mixing weights:")
            for name, weight in list(mixing_analysis.items())[:5]:  # Show top 5
                print(f"    {name:20}: {weight:.4f}")
            
            refinement_scale = filter_obj.refinement_scale.cpu().item()
            transform_scale = filter_obj.transform_scale.cpu().item()
            transform_bias = filter_obj.transform_bias.cpu().item()
            print(f"  Refinement scale: {refinement_scale:.4f}")
            print(f"  Transform scale:  {transform_scale:.4f}")
            print(f"  Transform bias:   {transform_bias:.4f}")
            
        elif isinstance(filter_obj, fl.AdaptiveGoldenFilter):
            adaptive_ratio = 0.36 + 0.1 * torch.tanh(filter_obj.golden_ratio_delta).cpu().item()
            print(f"  Learned golden ratio: {adaptive_ratio:.4f} (base: 0.36)")
            scale_factors = filter_obj.scale_factors.cpu().numpy()
            bias_terms = filter_obj.bias_terms.cpu().numpy()
            print(f"  Scale factors (first 3): {scale_factors[:3]}")
            print(f"  Bias terms (first 3): {bias_terms[:3]}")
            
        elif isinstance(filter_obj, fl.EigenvalueAdaptiveFilter):
            boundary_1 = torch.sigmoid(filter_obj.boundary_1).cpu().item() * 0.5
            boundary_2 = boundary_1 + torch.sigmoid(filter_obj.boundary_2).cpu().item() * 0.5
            print(f"  Learned boundaries: {boundary_1:.3f}, {boundary_2:.3f}")
            print(f"  Low freq coeffs:  {filter_obj.low_freq_coeffs.cpu().numpy()}")
            print(f"  Mid freq coeffs:  {filter_obj.mid_freq_coeffs.cpu().numpy()}")
            print(f"  High freq coeffs: {filter_obj.high_freq_coeffs.cpu().numpy()}")
            
        elif isinstance(filter_obj, fl.NeuralSpectralFilter):
            param_count = sum(p.numel() for p in filter_obj.parameters())
            print(f"  Neural filter - {param_count} parameters")
            print(f"  Network layers: {len([m for m in filter_obj.filter_net if isinstance(m, nn.Linear)])}")
            
        # NEW HIGH-CAPACITY FILTER DEBUGGING
        elif isinstance(filter_obj, fl.DeepSpectralFilter):
            param_count = sum(p.numel() for p in filter_obj.parameters())
            linear_layers = [m for m in filter_obj.filter_net if isinstance(m, nn.Linear)]
            print(f"  Deep filter - {param_count} parameters")
            print(f"  Architecture: {len(linear_layers)} linear layers")
            print(f"  Hidden dims: {[layer.out_features for layer in linear_layers[:-1]]}")
            
        elif isinstance(filter_obj, fl.MultiScaleSpectralFilter):
            param_count = sum(p.numel() for p in filter_obj.parameters())
            n_bands = filter_obj.n_bands
            boundaries = torch.sort(filter_obj.band_boundaries)[0].cpu().numpy()
            responses = torch.sigmoid(filter_obj.band_responses).cpu().numpy()
            sharpness = torch.abs(filter_obj.transition_sharpness).cpu().item()
            
            print(f"  MultiScale filter - {param_count} parameters")
            print(f"  Frequency bands: {n_bands}")
            print(f"  Boundaries: {boundaries[:5]}...")  # Show first 5
            print(f"  Responses: {responses[:5]}...")    # Show first 5
            print(f"  Transition sharpness: {sharpness:.2f}")
            
        elif isinstance(filter_obj, fl.EnsembleSpectralFilter):
            param_count = sum(p.numel() for p in filter_obj.parameters())
            ensemble_analysis = filter_obj.get_ensemble_analysis()
            
            print(f"  Ensemble filter - {param_count} parameters")
            print(f"  Component weights:")
            for name, weight in ensemble_analysis.items():
                if name != 'temperature':
                    print(f"    {name:12}: {weight:.4f}")
            print(f"  Temperature: {ensemble_analysis['temperature']:.4f}")
            
        else:
            # Fallback for unknown filter types
            param_count = sum(p.numel() for p in filter_obj.parameters())
            print(f"  Custom filter - {param_count} parameters")
            print(f"  Type: {type(filter_obj).__name__}")

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
    
    def get_parameter_count(self):
        """Get detailed parameter count breakdown"""
        total_params = sum(p.numel() for p in self.parameters())
        filter_params = sum(p.numel() for p in self.get_filter_parameters())
        other_params = total_params - filter_params
        
        return {
            'total': total_params,
            'filter': filter_params,
            'combination_weights': self.combination_weights.numel(),
            'other': other_params
        }