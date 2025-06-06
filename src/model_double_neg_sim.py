'''
Created on June 3, 2025
PyTorch Implementation of uSpec: Universal Spectral Collaborative Filtering
ENHANCED WITH MULTIPLE FILTER DESIGNS: Original, Basis, Adaptive, Neural, Deep, MultiScale, Ensemble

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import time
import gc
import world
import filters as fl


class UniversalSpectralCF(nn.Module):
    """
    Universal Spectral CF with Positive and Negative Similarities
    Enhanced with multiple filter designs from filters.py
    """
    def __init__(self, adj_mat, config=None):
        super().__init__()
        
        self.adj_mat = adj_mat
        self.config = config if config else {}
        self.device = self.config.get('device', 'cpu')
        self.n_eigen = self.config.get('n_eigen', 50)
        self.filter_order = self.config.get('filter_order', 6)
        self.lr = self.config.get('lr', 0.01)
        self.filter = self.config.get('filter', 'ui')
        
        # Filter design selection with new high-capacity options
        self.filter_design = self.config.get('filter_design', 'basis')
        self.init_filter = self.config.get('init_filter', 'smooth')
        
        print(f"Model Double - Filter Design: {self.filter_design}")
        print(f"Model Double - Initialization: {self.init_filter}")
        
        # Convert to tensor
        if sp.issparse(self.adj_mat):
            adj_dense = self.adj_mat.toarray()
        else:
            adj_dense = self.adj_mat
            
        self.register_buffer('adj_tensor', torch.tensor(adj_dense, dtype=torch.float32))
        self.n_users, self.n_items = self.adj_tensor.shape
        
        # Normalization for positive similarities
        row_sums = self.adj_tensor.sum(dim=1, keepdim=True) + 1e-8
        col_sums = self.adj_tensor.sum(dim=0, keepdim=True) + 1e-8
        self.register_buffer('norm_adj', self.adj_tensor / torch.sqrt(row_sums) / torch.sqrt(col_sums))
        
        # Create binary interaction matrix for negative similarities
        binary_adj = (self.adj_tensor > 0).float()
        complement_adj = 1 - binary_adj
        
        # Normalize complement matrix for negative similarities
        neg_row_sums = complement_adj.sum(dim=1, keepdim=True) + 1e-8
        neg_col_sums = complement_adj.sum(dim=0, keepdim=True) + 1e-8
        self.register_buffer('neg_norm_adj', complement_adj / torch.sqrt(neg_row_sums) / torch.sqrt(neg_col_sums))
        
        # Enhanced filter modules (will be set in _initialize_model)
        self.user_pos_filter = None
        self.user_neg_filter = None
        self.item_pos_filter = None
        self.item_neg_filter = None
        
        # Clean up intermediate variables
        del adj_dense, row_sums, col_sums, binary_adj, complement_adj, neg_row_sums, neg_col_sums
        self._memory_cleanup()
        
        # Initialize the model
        self._initialize_model()
    
    def _memory_cleanup(self):
        """Force memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _create_filter_from_design(self):
        """Create filter based on selected design (UPDATED with new filters)"""
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
    
    def _initialize_model(self):
        """Initialize eigendecompositions and enhanced filters for both positive and negative similarities"""
        start = time.time()
        
        print(f"Computing positive and negative eigendecompositions for filter type: {self.filter}")
        
        # Compute positive similarity matrices
        print("Computing positive similarity matrices...")
        user_pos_sim = self._compute_similarity_chunked(self.norm_adj, self.norm_adj.t(), chunk_size=1000)
        item_pos_sim = self._compute_similarity_chunked(self.norm_adj.t(), self.norm_adj, chunk_size=1000)
        
        # Compute negative similarity matrices
        print("Computing negative similarity matrices...")
        user_neg_sim = self._compute_similarity_chunked(self.neg_norm_adj, self.neg_norm_adj.t(), chunk_size=1000)
        item_neg_sim = self._compute_similarity_chunked(self.neg_norm_adj.t(), self.neg_norm_adj, chunk_size=1000)
        
        # Initialize positive and negative filters for users
        if self.filter in ['u', 'ui']:
            print("Processing user positive similarities...")
            # Positive user similarities
            user_pos_sim_np = user_pos_sim.cpu().numpy()
            k_user = min(self.n_eigen, self.n_users - 2)
            try:
                eigenvals, eigenvecs = eigsh(sp.csr_matrix(user_pos_sim_np), k=k_user, which='LM')
                self.register_buffer('user_pos_eigenvals', torch.tensor(np.real(eigenvals), dtype=torch.float32))
                self.register_buffer('user_pos_eigenvecs', torch.tensor(np.real(eigenvecs), dtype=torch.float32))
                self.user_pos_filter = self._create_filter_from_design()
                print(f"User positive eigendecomposition: {k_user} components")
            except Exception as e:
                print(f"User positive eigendecomposition failed: {e}")
                self.register_buffer('user_pos_eigenvals', torch.ones(k_user))
                self.register_buffer('user_pos_eigenvecs', torch.eye(self.n_users, k_user))
                self.user_pos_filter = self._create_filter_from_design()
            
            del user_pos_sim_np
            self._memory_cleanup()
            
            print("Processing user negative similarities...")
            # Negative user similarities
            user_neg_sim_np = user_neg_sim.cpu().numpy()
            try:
                eigenvals, eigenvecs = eigsh(sp.csr_matrix(user_neg_sim_np), k=k_user, which='LM')
                self.register_buffer('user_neg_eigenvals', torch.tensor(np.real(eigenvals), dtype=torch.float32))
                self.register_buffer('user_neg_eigenvecs', torch.tensor(np.real(eigenvecs), dtype=torch.float32))
                self.user_neg_filter = self._create_filter_from_design()
                print(f"User negative eigendecomposition: {k_user} components")
            except Exception as e:
                print(f"User negative eigendecomposition failed: {e}")
                self.register_buffer('user_neg_eigenvals', torch.ones(k_user))
                self.register_buffer('user_neg_eigenvecs', torch.eye(self.n_users, k_user))
                self.user_neg_filter = self._create_filter_from_design()
            
            del user_neg_sim_np
            self._memory_cleanup()
        
        # Initialize positive and negative filters for items
        if self.filter in ['i', 'ui']:
            print("Processing item positive similarities...")
            # Positive item similarities
            item_pos_sim_np = item_pos_sim.cpu().numpy()
            k_item = min(self.n_eigen, self.n_items - 2)
            try:
                eigenvals, eigenvecs = eigsh(sp.csr_matrix(item_pos_sim_np), k=k_item, which='LM')
                self.register_buffer('item_pos_eigenvals', torch.tensor(np.real(eigenvals), dtype=torch.float32))
                self.register_buffer('item_pos_eigenvecs', torch.tensor(np.real(eigenvecs), dtype=torch.float32))
                self.item_pos_filter = self._create_filter_from_design()
                print(f"Item positive eigendecomposition: {k_item} components")
            except Exception as e:
                print(f"Item positive eigendecomposition failed: {e}")
                self.register_buffer('item_pos_eigenvals', torch.ones(k_item))
                self.register_buffer('item_pos_eigenvecs', torch.eye(self.n_items, k_item))
                self.item_pos_filter = self._create_filter_from_design()
            
            del item_pos_sim_np
            self._memory_cleanup()
            
            print("Processing item negative similarities...")
            # Negative item similarities
            item_neg_sim_np = item_neg_sim.cpu().numpy()
            try:
                eigenvals, eigenvecs = eigsh(sp.csr_matrix(item_neg_sim_np), k=k_item, which='LM')
                self.register_buffer('item_neg_eigenvals', torch.tensor(np.real(eigenvals), dtype=torch.float32))
                self.register_buffer('item_neg_eigenvecs', torch.tensor(np.real(eigenvecs), dtype=torch.float32))
                self.item_neg_filter = self._create_filter_from_design()
                print(f"Item negative eigendecomposition: {k_item} components")
            except Exception as e:
                print(f"Item negative eigendecomposition failed: {e}")
                self.register_buffer('item_neg_eigenvals', torch.ones(k_item))
                self.register_buffer('item_neg_eigenvecs', torch.eye(self.n_items, k_item))
                self.item_neg_filter = self._create_filter_from_design()
            
            del item_neg_sim_np
            self._memory_cleanup()
        
        # Clean up similarity matrices
        del user_pos_sim, item_pos_sim, user_neg_sim, item_neg_sim
        self._memory_cleanup()
        
        # Set combination weights as nn.Parameter (trainable)
        if self.filter == 'u':
            # direct + user_pos + user_neg
            self.combination_weights = nn.Parameter(torch.tensor([0.4, 0.3, 0.3]))
        elif self.filter == 'i':
            # direct + item_pos + item_neg
            self.combination_weights = nn.Parameter(torch.tensor([0.4, 0.3, 0.3]))
        else:  # 'ui'
            # direct + item_pos + item_neg + user_pos + user_neg
            self.combination_weights = nn.Parameter(torch.tensor([0.3, 0.2, 0.2, 0.15, 0.15]))
        
        end = time.time()
        print(f'Enhanced initialization time for Universal-SpectralCF with pos/neg similarities ({self.filter}): {end-start:.2f}s')
    
    def _get_filter_matrices(self):
        """Compute filter matrices for both positive and negative similarities"""
        user_pos_matrix = None
        user_neg_matrix = None
        item_pos_matrix = None
        item_neg_matrix = None
        
        if self.filter in ['u', 'ui']:
            if self.user_pos_filter is not None:
                user_pos_filter_response = self.user_pos_filter(self.user_pos_eigenvals)
                user_pos_matrix = self.user_pos_eigenvecs @ torch.diag(user_pos_filter_response) @ self.user_pos_eigenvecs.t()
            
            if self.user_neg_filter is not None:
                user_neg_filter_response = self.user_neg_filter(self.user_neg_eigenvals)
                user_neg_matrix = self.user_neg_eigenvecs @ torch.diag(user_neg_filter_response) @ self.user_neg_eigenvecs.t()
        
        if self.filter in ['i', 'ui']:
            if self.item_pos_filter is not None:
                item_pos_filter_response = self.item_pos_filter(self.item_pos_eigenvals)
                item_pos_matrix = self.item_pos_eigenvecs @ torch.diag(item_pos_filter_response) @ self.item_pos_eigenvecs.t()
            
            if self.item_neg_filter is not None:
                item_neg_filter_response = self.item_neg_filter(self.item_neg_eigenvals)
                item_neg_matrix = self.item_neg_eigenvecs @ torch.diag(item_neg_filter_response) @ self.item_neg_eigenvecs.t()
        
        return user_pos_matrix, user_neg_matrix, item_pos_matrix, item_neg_matrix
    
    def forward(self, users):
        """Clean forward pass - ONLY returns predictions"""
        user_profiles = self.adj_tensor[users]
        direct_scores = user_profiles
        
        # Get filter matrices
        user_pos_matrix, user_neg_matrix, item_pos_matrix, item_neg_matrix = self._get_filter_matrices()
        
        # Compute scores based on filter type
        if self.filter == 'u':
            # User-based filtering with positive and negative similarities
            user_pos_filter_rows = user_pos_matrix[users]
            user_pos_scores = user_pos_filter_rows @ self.adj_tensor
            
            user_neg_filter_rows = user_neg_matrix[users]
            user_neg_scores = user_neg_filter_rows @ self.neg_norm_adj
            
            weights = torch.softmax(self.combination_weights, dim=0)
            predicted = weights[0] * direct_scores + weights[1] * user_pos_scores + weights[2] * user_neg_scores
            
        elif self.filter == 'i':
            # Item-based filtering with positive and negative similarities
            item_pos_scores = user_profiles @ item_pos_matrix
            item_neg_scores = user_profiles @ item_neg_matrix
            
            weights = torch.softmax(self.combination_weights, dim=0)
            predicted = weights[0] * direct_scores + weights[1] * item_pos_scores + weights[2] * item_neg_scores
            
        else:  # 'ui'
            # Combined user and item filtering with positive and negative similarities
            item_pos_scores = user_profiles @ item_pos_matrix
            item_neg_scores = user_profiles @ item_neg_matrix
            
            user_pos_filter_rows = user_pos_matrix[users]
            user_pos_scores = user_pos_filter_rows @ self.adj_tensor
            
            user_neg_filter_rows = user_neg_matrix[users]
            user_neg_scores = user_neg_filter_rows @ self.neg_norm_adj
            
            weights = torch.softmax(self.combination_weights, dim=0)
            predicted = (weights[0] * direct_scores + 
                        weights[1] * item_pos_scores + 
                        weights[2] * item_neg_scores + 
                        weights[3] * user_pos_scores + 
                        weights[4] * user_neg_scores)
        
        # Memory cleanup for large datasets
        if self.training and (self.n_users > 10000 or self.n_items > 10000):
            del user_pos_matrix, user_neg_matrix, item_pos_matrix, item_neg_matrix
            self._memory_cleanup()
        
        return predicted  # ALWAYS return predictions only!
        
    def getUsersRating(self, batch_users):
        """Memory-efficient evaluation interface"""
        self.eval()
        with torch.no_grad():
            if isinstance(batch_users, np.ndarray):
                batch_users = torch.LongTensor(batch_users)
            
            # Use simplified forward pass
            combined = self.forward(batch_users)
            result = combined.cpu().numpy()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return result
    
    def get_filter_parameters(self):
        """Get filter parameters for separate optimization"""
        filter_params = []
        if self.user_pos_filter is not None:
            filter_params.extend(self.user_pos_filter.parameters())
        if self.user_neg_filter is not None:
            filter_params.extend(self.user_neg_filter.parameters())
        if self.item_pos_filter is not None:
            filter_params.extend(self.item_pos_filter.parameters())
        if self.item_neg_filter is not None:
            filter_params.extend(self.item_neg_filter.parameters())
        return filter_params
    
    def get_other_parameters(self):
        """Get non-filter parameters"""
        filter_param_ids = {id(p) for p in self.get_filter_parameters()}
        return [p for p in self.parameters() if id(p) not in filter_param_ids]

    def debug_filter_learning(self):
        """Enhanced debug for different filter designs with pos/neg similarities"""
        print(f"\n=== FILTER LEARNING DEBUG (POS/NEG) - {self.filter_design.upper()} ===")
        
        with torch.no_grad():
            if self.filter in ['u', 'ui']:
                if self.user_pos_filter is not None:
                    print(f"\n👤 User Positive Filter ({self.filter_design}):")
                    self._debug_single_filter(self.user_pos_filter, "User Positive")
                
                if self.user_neg_filter is not None:
                    print(f"\n👤 User Negative Filter ({self.filter_design}):")
                    self._debug_single_filter(self.user_neg_filter, "User Negative")
            
            if self.filter in ['i', 'ui']:
                if self.item_pos_filter is not None:
                    print(f"\n🎬 Item Positive Filter ({self.filter_design}):")
                    self._debug_single_filter(self.item_pos_filter, "Item Positive")
                
                if self.item_neg_filter is not None:
                    print(f"\n🎬 Item Negative Filter ({self.filter_design}):")
                    self._debug_single_filter(self.item_neg_filter, "Item Negative")
            
            # Combination weights
            weights = torch.softmax(self.combination_weights, dim=0)
            weight_names = {
                'u': ['Direct', 'User+', 'User-'],
                'i': ['Direct', 'Item+', 'Item-'],
                'ui': ['Direct', 'Item+', 'Item-', 'User+', 'User-']
            }
            print(f"\n🔗 Combination Weights:")
            for i, (name, weight) in enumerate(zip(weight_names[self.filter], weights)):
                print(f"  {name:8}: {weight.cpu().item():.4f}")
        
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