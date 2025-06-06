'''
Created on June 3, 2025
PyTorch Implementation of uSpec: Universal Spectral Collaborative Filtering
DySimGCF-Style Implementation with True Similarity-Based Graph Construction

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.metrics.pairwise import cosine_similarity
import time
import gc
import world
import filters as fl


class UniversalSpectralCF(nn.Module):
    """
    Universal Spectral CF with DySimGCF-Style Similarity-Based Graph Construction
    Uses the same similarity computation approach as DySimGCF but applies spectral filtering
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
        
        # DySimGCF-style parameters (matching their configuration)
        self.u_sim = self.config.get('u_sim', 'cos')           # 'cos' or 'jac'
        self.i_sim = self.config.get('i_sim', 'cos')           # 'cos' or 'jac'
        self.u_K = self.config.get('u_K', 80)                 # Top-K users (DySimGCF default)
        self.i_K = self.config.get('i_K', 10)                 # Top-K items (DySimGCF default)
        self.self_loop = self.config.get('self_loop', False)   # Self-loops in similarity
        
        # Filter design selection
        self.filter_design = self.config.get('filter_design', 'basis')
        self.init_filter = self.config.get('init_filter', 'smooth')
        
        print(f"Model Double (DySimGCF-Style) - Filter Design: {self.filter_design}")
        print(f"Model Double (DySimGCF-Style) - Initialization: {self.init_filter}")
        print(f"Model Double (DySimGCF-Style) - User Sim: {self.u_sim} (K={self.u_K}), Item Sim: {self.i_sim} (K={self.i_K})")
        print(f"Model Double (DySimGCF-Style) - Self-loop: {self.self_loop}")
        
        # Convert to tensor
        if sp.issparse(self.adj_mat):
            adj_dense = self.adj_mat.toarray()
        else:
            adj_dense = self.adj_mat
            
        self.register_buffer('adj_tensor', torch.tensor(adj_dense, dtype=torch.float32))
        self.n_users, self.n_items = self.adj_tensor.shape
        
        # Create DySimGCF-style similarity matrices using their exact approach
        self._create_dysimgcf_similarity_matrices()
        
        # Filter modules (will be set in _initialize_model)
        self.user_filter = None
        self.item_filter = None
        
        # Clean up intermediate variables
        del adj_dense
        self._memory_cleanup()
        
        # Initialize the model
        self._initialize_model()
    
    def _memory_cleanup(self):
        """Force memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _dysimgcf_cosine_similarity(self, matrix, top_k, self_loop=False):
        """
        DySimGCF-style cosine similarity computation
        Replicates the exact approach from i_sim.py
        """
        print(f"  Computing DySimGCF cosine similarity (top-K={top_k})...")
        
        # Convert to sparse matrix for DySimGCF compatibility
        from scipy.sparse import csr_matrix
        
        # Convert to binary sparse matrix (DySimGCF approach)
        binary_matrix = csr_matrix((matrix > 0).astype(int))
        
        # Compute cosine similarity - force sparse output
        try:
            similarity_matrix = cosine_similarity(binary_matrix, dense_output=False)
        except:
            # Fallback: compute dense then convert to sparse
            similarity_dense = cosine_similarity(binary_matrix.toarray())
            similarity_matrix = csr_matrix(similarity_dense)
        
        # Handle different return types from sklearn
        if hasattr(similarity_matrix, 'setdiag'):
            # It's a sparse matrix
            if self_loop:
                similarity_matrix.setdiag(1)
            else:
                similarity_matrix.setdiag(0)
        else:
            # It's a dense array, convert to sparse
            if not self_loop:
                np.fill_diagonal(similarity_matrix, 0)
            else:
                np.fill_diagonal(similarity_matrix, 1)
            similarity_matrix = csr_matrix(similarity_matrix)
        
        # DySimGCF-style top-K filtering
        filtered_data = []
        filtered_rows = []
        filtered_cols = []
        
        for i in range(similarity_matrix.shape[0]):
            # Get the non-zero elements in the i-th row
            if hasattr(similarity_matrix, 'getrow'):
                row = similarity_matrix.getrow(i).tocoo()
                if row.nnz == 0:
                    continue
                row_data = row.data
                row_indices = row.col
            else:
                # Handle dense case
                row_data = similarity_matrix[i]
                nonzero_mask = row_data != 0
                if not np.any(nonzero_mask):
                    continue
                row_data = row_data[nonzero_mask]
                row_indices = np.where(nonzero_mask)[0]
            
            # Sort and select top-K (DySimGCF approach)
            if len(row_data) > top_k:
                top_k_idx = np.argsort(-row_data)[:top_k]
            else:
                top_k_idx = np.argsort(-row_data)
            
            # Store the top-K similarities
            filtered_data.extend(row_data[top_k_idx])
            filtered_rows.extend([i] * len(top_k_idx))
            filtered_cols.extend(row_indices[top_k_idx])
        
        # Create filtered sparse matrix
        from scipy.sparse import coo_matrix
        filtered_similarity_matrix = coo_matrix(
            (filtered_data, (filtered_rows, filtered_cols)), 
            shape=similarity_matrix.shape
        )
        
        return filtered_similarity_matrix.tocsr()
    
    def _dysimgcf_jaccard_similarity(self, matrix, top_k, self_loop=False):
        """
        DySimGCF-style Jaccard similarity computation
        Replicates the exact approach from i_sim.py
        """
        print(f"  Computing DySimGCF Jaccard similarity (top-K={top_k})...")
        
        # Convert to sparse matrix for efficiency
        from scipy.sparse import csr_matrix
        
        # Convert to binary matrix (ensure sparse)
        if not sp.issparse(matrix):
            binary_matrix = csr_matrix((matrix > 0).astype(int))
        else:
            binary_matrix = csr_matrix((matrix > 0).astype(int))
        
        # Compute intersection using dot product (DySimGCF approach)
        intersection = binary_matrix.dot(binary_matrix.T)
        
        # Compute row sums
        row_sums = np.array(binary_matrix.sum(axis=1)).flatten()
        
        # Compute union - convert intersection to dense for computation
        intersection_dense = intersection.toarray().astype(np.float32)
        union = row_sums[:, None] + row_sums[None, :] - intersection_dense
        
        # Compute Jaccard similarity
        similarity_matrix = np.divide(
            intersection_dense, union, 
            out=np.zeros_like(intersection_dense, dtype=np.float32), 
            where=union != 0
        )
        
        # Handle self-loops
        if self_loop:
            np.fill_diagonal(similarity_matrix, 1)
        else:
            np.fill_diagonal(similarity_matrix, 0)
        
        # DySimGCF-style top-K filtering
        filtered_data = []
        filtered_rows = []
        filtered_cols = []
        
        for i in range(similarity_matrix.shape[0]):
            row = similarity_matrix[i]
            if np.count_nonzero(row) == 0:
                continue
            
            # Sort and select top-K
            top_k_idx = np.argsort(-row)[:top_k]
            
            # Store the top-K similarities
            filtered_data.extend(row[top_k_idx])
            filtered_rows.extend([i] * len(top_k_idx))
            filtered_cols.extend(top_k_idx)
        
        # Create filtered sparse matrix
        from scipy.sparse import coo_matrix
        filtered_similarity_matrix = coo_matrix(
            (filtered_data, (filtered_rows, filtered_cols)), 
            shape=similarity_matrix.shape
        )
        
        return filtered_similarity_matrix.tocsr()
    
    def _create_dysimgcf_similarity_matrices(self):
        """
        Create DySimGCF-style similarity matrices - keep them separate for efficiency
        """
        print("Creating DySimGCF-style similarity matrices...")
        
        # Convert to numpy for similarity computation (DySimGCF approach)
        user_item_matrix = self.adj_tensor.cpu().numpy()
        
        # User-user similarity using DySimGCF approach
        print(f"  User-user similarity ({self.u_sim})...")
        if self.u_sim == 'cos':
            user_similarity_matrix = self._dysimgcf_cosine_similarity(
                user_item_matrix, self.u_K, self.self_loop
            )
        elif self.u_sim == 'jac':
            user_similarity_matrix = self._dysimgcf_jaccard_similarity(
                user_item_matrix, self.u_K, self.self_loop
            )
        else:
            raise ValueError(f"Unsupported user similarity type: {self.u_sim}")
        
        # Item-item similarity using DySimGCF approach
        print(f"  Item-item similarity ({self.i_sim})...")
        if self.i_sim == 'cos':
            item_similarity_matrix = self._dysimgcf_cosine_similarity(
                user_item_matrix.T, self.i_K, self.self_loop
            )
        elif self.i_sim == 'jac':
            item_similarity_matrix = self._dysimgcf_jaccard_similarity(
                user_item_matrix.T, self.i_K, self.self_loop
            )
        else:
            raise ValueError(f"Unsupported item similarity type: {self.i_sim}")
        
        print(f"    User similarity: {user_similarity_matrix.nnz} edges, "
              f"avg degree: {user_similarity_matrix.nnz / self.n_users:.1f}")
        print(f"    Item similarity: {item_similarity_matrix.nnz} edges, "
              f"avg degree: {item_similarity_matrix.nnz / self.n_items:.1f}")
        
        # Store similarity matrices separately (more memory efficient)
        self.register_buffer('user_sim_adj', 
                           torch.tensor(user_similarity_matrix.toarray(), dtype=torch.float32))
        self.register_buffer('item_sim_adj', 
                           torch.tensor(item_similarity_matrix.toarray(), dtype=torch.float32))
        
        # Clean up
        del user_item_matrix, user_similarity_matrix, item_similarity_matrix
        self._memory_cleanup()
    
    def _create_filter_from_design(self):
        """Create filter based on selected design"""
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
        """Initialize eigendecompositions using DySimGCF-style similarity matrices"""
        start = time.time()
        
        print(f"Computing DySimGCF-style eigendecompositions for filter type: {self.filter}")
        
        # User eigendecomposition (on DySimGCF similarity matrix)
        if self.filter in ['u', 'ui']:
            print("Processing user DySimGCF similarity matrix...")
            
            # Use the precomputed DySimGCF similarity matrix directly
            user_sim_np = self.user_sim_adj.cpu().numpy()
            k_user = min(self.n_eigen, self.n_users - 2)
            
            try:
                eigenvals, eigenvecs = eigsh(sp.csr_matrix(user_sim_np), k=k_user, which='LM')
                self.register_buffer('user_eigenvals', 
                                   torch.tensor(np.real(eigenvals), dtype=torch.float32))
                self.register_buffer('user_eigenvecs', 
                                   torch.tensor(np.real(eigenvecs), dtype=torch.float32))
                self.user_filter = self._create_filter_from_design()
                print(f"  User eigendecomposition: {k_user} components, max eigenval = {eigenvals.max():.4f}")
            except Exception as e:
                print(f"  User eigendecomposition failed: {e}")
                self.register_buffer('user_eigenvals', torch.ones(k_user))
                self.register_buffer('user_eigenvecs', torch.eye(self.n_users, k_user))
                self.user_filter = self._create_filter_from_design()
            
            del user_sim_np
            self._memory_cleanup()
        
        # Item eigendecomposition (on DySimGCF similarity matrix)
        if self.filter in ['i', 'ui']:
            print("Processing item DySimGCF similarity matrix...")
            
            # Use the precomputed DySimGCF similarity matrix directly
            item_sim_np = self.item_sim_adj.cpu().numpy()
            k_item = min(self.n_eigen, self.n_items - 2)
            
            try:
                eigenvals, eigenvecs = eigsh(sp.csr_matrix(item_sim_np), k=k_item, which='LM')
                self.register_buffer('item_eigenvals', 
                                   torch.tensor(np.real(eigenvals), dtype=torch.float32))
                self.register_buffer('item_eigenvecs', 
                                   torch.tensor(np.real(eigenvecs), dtype=torch.float32))
                self.item_filter = self._create_filter_from_design()
                print(f"  Item eigendecomposition: {k_item} components, max eigenval = {eigenvals.max():.4f}")
            except Exception as e:
                print(f"  Item eigendecomposition failed: {e}")
                self.register_buffer('item_eigenvals', torch.ones(k_item))
                self.register_buffer('item_eigenvecs', torch.eye(self.n_items, k_item))
                self.item_filter = self._create_filter_from_design()
            
            del item_sim_np
            self._memory_cleanup()
        
        # Set combination weights as nn.Parameter (trainable)
        if self.filter == 'u':
            # direct + user_sim  
            self.combination_weights = nn.Parameter(torch.tensor([0.6, 0.4]))
        elif self.filter == 'i':
            # direct + item_sim
            self.combination_weights = nn.Parameter(torch.tensor([0.6, 0.4]))
        else:  # 'ui'
            # direct + item_sim + user_sim
            self.combination_weights = nn.Parameter(torch.tensor([0.5, 0.3, 0.2]))
        
        end = time.time()
        print(f'DySimGCF-style initialization time for Universal-SpectralCF ({self.filter}): {end-start:.2f}s')
    
    def _get_filter_matrices(self):
        """Compute filter matrices for DySimGCF similarity-based adjacencies"""
        user_matrix = None
        item_matrix = None
        
        if self.filter in ['u', 'ui'] and self.user_filter is not None:
            user_filter_response = self.user_filter(self.user_eigenvals)
            user_matrix = self.user_eigenvecs @ torch.diag(user_filter_response) @ self.user_eigenvecs.t()
        
        if self.filter in ['i', 'ui'] and self.item_filter is not None:
            item_filter_response = self.item_filter(self.item_eigenvals)
            item_matrix = self.item_eigenvecs @ torch.diag(item_filter_response) @ self.item_eigenvecs.t()
        
        return user_matrix, item_matrix
    
    def forward(self, users):
        """Clean forward pass - ONLY returns predictions"""
        user_profiles = self.adj_tensor[users]
        direct_scores = user_profiles
        
        # Get filter matrices
        user_matrix, item_matrix = self._get_filter_matrices()
        
        # Compute scores based on filter type
        if self.filter == 'u':
            # User-based filtering with DySimGCF-style similarity
            user_filter_rows = user_matrix[users]
            user_scores = user_filter_rows @ self.adj_tensor
            
            weights = torch.softmax(self.combination_weights, dim=0)
            predicted = weights[0] * direct_scores + weights[1] * user_scores
            
        elif self.filter == 'i':
            # Item-based filtering with DySimGCF-style similarity
            item_scores = user_profiles @ item_matrix
            
            weights = torch.softmax(self.combination_weights, dim=0)
            predicted = weights[0] * direct_scores + weights[1] * item_scores
            
        else:  # 'ui'
            # Combined user and item filtering with DySimGCF-style similarities
            item_scores = user_profiles @ item_matrix
            user_filter_rows = user_matrix[users]
            user_scores = user_filter_rows @ self.adj_tensor
            
            weights = torch.softmax(self.combination_weights, dim=0)
            predicted = (weights[0] * direct_scores + 
                        weights[1] * item_scores + 
                        weights[2] * user_scores)
        
        # Memory cleanup for large datasets
        if self.training and (self.n_users > 10000 or self.n_items > 10000):
            del user_matrix, item_matrix
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
        """Enhanced debug for different filter designs with DySimGCF-style similarities"""
        print(f"\n=== FILTER LEARNING DEBUG (DYSIMGCF-STYLE) - {self.filter_design.upper()} ===")
        print(f"User Sim: {self.u_sim} (K={self.u_K}), Item Sim: {self.i_sim} (K={self.i_K}), Self-loop: {self.self_loop}")
        
        with torch.no_grad():
            if self.filter in ['u', 'ui'] and self.user_filter is not None:
                print(f"\n👤 User DySimGCF Filter ({self.filter_design}):")
                self._debug_single_filter(self.user_filter, "User")
            
            if self.filter in ['i', 'ui'] and self.item_filter is not None:
                print(f"\n🎬 Item DySimGCF Filter ({self.filter_design}):")
                self._debug_single_filter(self.item_filter, "Item")
            
            # Combination weights
            weights = torch.softmax(self.combination_weights, dim=0)
            weight_names = {
                'u': ['Direct', 'UserSim'],
                'i': ['Direct', 'ItemSim'],
                'ui': ['Direct', 'ItemSim', 'UserSim']
            }
            print(f"\n🔗 Combination Weights:")
            for i, (name, weight) in enumerate(zip(weight_names[self.filter], weights)):
                print(f"  {name:10}: {weight.cpu().item():.4f}")
        
        print("=== END DEBUG ===\n")
    
    def _debug_single_filter(self, filter_obj, filter_name):
        """Debug individual filter based on its type"""
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