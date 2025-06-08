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
        
        # DySimGCF-style parameters (with conservative defaults for better performance)
        self.u_sim = self.config.get('u_sim', 'cos')           # 'cos' or 'jac'
        self.i_sim = self.config.get('i_sim', 'cos')           # 'cos' or 'jac'
        
        # Conservative adaptive filtering (but not too dense)
        self.adaptive = self.config.get('adaptive', True)      # Use adaptive filtering
        self.u_threshold = self.config.get('u_threshold', 0.1) # Moderate threshold (avoid too dense)
        self.i_threshold = self.config.get('i_threshold', 0.1) # Moderate threshold (avoid too dense)
        self.u_K = self.config.get('u_K', 30)                 # Moderate K (not too many connections)
        self.i_K = self.config.get('i_K', 20)                 # Moderate K (not too many connections)
        
        # More conservative adaptive parameters
        self.min_threshold = self.config.get('min_threshold', 0.05)  # Reasonable min
        self.max_threshold = self.config.get('max_threshold', 0.25)  # Reasonable max
        self.use_topk = self.config.get('use_topk', True)            # Use top-K to control density
        
        # Add direct adjacency option
        self.use_direct = self.config.get('use_direct', True)        # Include direct connections
            
        self.self_loop = self.config.get('self_loop', False)   # Self-loops in similarity
        
        # Filter design selection
        self.filter_design = self.config.get('filter_design', 'basis')
        self.init_filter = self.config.get('init_filter', 'smooth')
        
        print(f"Model Double (DySimGCF-Style) - Filter Design: {self.filter_design}")
        print(f"Model Double (DySimGCF-Style) - Initialization: {self.init_filter}")
        if self.adaptive:
            if self.use_topk:
                print(f"Model Double (DySimGCF-Style) - Adaptive Top-K: User {self.u_sim} (base-K={self.u_K}), Item {self.i_sim} (base-K={self.i_K})")
            else:
                print(f"Model Double (DySimGCF-Style) - Adaptive Threshold: User {self.u_sim} (base={self.u_threshold}), Item {self.i_sim} (base={self.i_threshold})")
        else:
            if self.use_topk:
                print(f"Model Double (DySimGCF-Style) - Fixed Top-K: User {self.u_sim} (K={self.u_K}), Item {self.i_sim} (K={self.i_K})")
            else:
                print(f"Model Double (DySimGCF-Style) - Fixed Threshold: User {self.u_sim} (≥{self.u_threshold}), Item {self.i_sim} (≥{self.i_threshold})")
        print(f"Model Double (DySimGCF-Style) - Eigenvalues: {self.n_eigen}, Self-loop: {self.self_loop}")
        
        # Convert to tensor
        if sp.issparse(self.adj_mat):
            adj_dense = self.adj_mat.toarray()
        else:
            adj_dense = self.adj_mat
            
        self.register_buffer('adj_tensor', torch.tensor(adj_dense, dtype=torch.float32))
        self.n_users, self.n_items = self.adj_tensor.shape
        
        # Create DySimGCF-style similarity matrices using their exact approach
        self._create_dysimgcf_similarity_matrices()
        
        # Filter modules (will be set in _initialize_model_enhanced)
        self.direct_user_filter = None
        self.direct_item_filter = None
        self.sim_user_filter = None
        self.sim_item_filter = None
        
        # Clean up intermediate variables
        del adj_dense
        self._memory_cleanup()
        
        # Create DySimGCF-style similarity matrices
        self._create_dysimgcf_similarity_matrices()
        
        # Initialize the model with both direct and similarity-based approaches
        self._initialize_model_enhanced()
    
    def _memory_cleanup(self):
        """Force memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _adaptive_cosine_similarity(self, matrix, base_threshold_or_k, base_k, self_loop=False, entity_type='user'):
        """
        Adaptive cosine similarity - adjusts filtering based on interaction patterns
        Replicates DySimGCF's adaptive approach from i_sim.py
        """
        from scipy.sparse import csr_matrix
        from tqdm import tqdm
        
        print(f"    Computing adaptive cosine similarity ({entity_type})...")
        
        # Convert to binary sparse matrix
        binary_matrix = csr_matrix((matrix > 0).astype(int))
        
        # Calculate interaction counts for adaptive filtering
        interaction_counts = np.array(binary_matrix.sum(axis=1)).flatten()
        avg_interaction_count = max(1, np.mean(interaction_counts))
        
        print(f"    Interaction stats: mean={avg_interaction_count:.1f}, "
              f"min={interaction_counts.min()}, max={interaction_counts.max()}")
        
        # Compute full cosine similarity matrix
        try:
            similarity_matrix = cosine_similarity(binary_matrix, dense_output=False)
        except:
            similarity_matrix = cosine_similarity(binary_matrix.toarray())
            similarity_matrix = csr_matrix(similarity_matrix)
        
        # Handle self-loops
        if self_loop:
            similarity_matrix.setdiag(1)
        else:
            similarity_matrix.setdiag(0)
        
        if self.use_topk:
            # Adaptive top-K approach (like DySimGCF)
            return self._adaptive_topk_filter(similarity_matrix, base_k, interaction_counts, avg_interaction_count, entity_type)
        else:
            # Adaptive threshold approach (our enhancement)
            return self._adaptive_threshold_filter(similarity_matrix, base_threshold_or_k, interaction_counts, avg_interaction_count, entity_type)
    
    def _adaptive_jaccard_similarity(self, matrix, base_threshold_or_k, base_k, self_loop=False, entity_type='user'):
        """
        Adaptive Jaccard similarity - adjusts filtering based on interaction patterns
        """
        from scipy.sparse import csr_matrix
        
        print(f"    Computing adaptive Jaccard similarity ({entity_type})...")
        
        # Convert to binary matrix
        binary_matrix = csr_matrix((matrix > 0).astype(int))
        
        # Calculate interaction counts for adaptive filtering
        interaction_counts = np.array(binary_matrix.sum(axis=1)).flatten()
        avg_interaction_count = max(1, np.mean(interaction_counts))
        
        print(f"    Interaction stats: mean={avg_interaction_count:.1f}, "
              f"min={interaction_counts.min()}, max={interaction_counts.max()}")
        
        # Compute intersection using dot product
        intersection = binary_matrix.dot(binary_matrix.T)
        
        # Compute row sums
        row_sums = np.array(binary_matrix.sum(axis=1)).flatten()
        
        # Compute union
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
            np.fill_diagonal(similarity_matrix, 1.0)
        else:
            np.fill_diagonal(similarity_matrix, 0.0)
        
        # Convert to sparse for filtering
        similarity_matrix = csr_matrix(similarity_matrix)
        
        if self.use_topk:
            # Adaptive top-K approach
            return self._adaptive_topk_filter(similarity_matrix, base_k, interaction_counts, avg_interaction_count, entity_type)
        else:
            # Adaptive threshold approach
            return self._adaptive_threshold_filter(similarity_matrix, base_threshold_or_k, interaction_counts, avg_interaction_count, entity_type)
    
    def _adaptive_topk_filter(self, similarity_matrix, base_k, interaction_counts, avg_interaction_count, entity_type):
        """
        Adaptive top-K filtering (like DySimGCF's approach)
        """
        from scipy.sparse import coo_matrix
        from tqdm import tqdm
        
        filtered_data = []
        filtered_rows = []
        filtered_cols = []
        
        min_k = 5 if entity_type == 'item' else 10
        max_k = 50 if entity_type == 'item' else 100
        
        desc = f'Adaptive top-K {entity_type} similarity (base-K={base_k})'
        pbar = tqdm(range(similarity_matrix.shape[0]), desc=desc, 
                   bar_format='{desc}{bar:30} {percentage:3.0f}% | {elapsed}')
        
        for i in pbar:
            # Calculate adaptive K based on square root scaling (DySimGCF approach)
            current_k = int(base_k * np.sqrt(interaction_counts[i]) / np.sqrt(avg_interaction_count))
            current_k = max(min_k, min(current_k, max_k))
            
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
            
            # Sort and select top-K
            if len(row_data) > current_k:
                top_k_idx = np.argsort(-row_data)[:current_k]
            else:
                top_k_idx = np.argsort(-row_data)
            
            # Store the top-K similarities
            filtered_data.extend(row_data[top_k_idx])
            filtered_rows.extend([i] * len(top_k_idx))
            filtered_cols.extend(row_indices[top_k_idx])
        
        # Create symmetric sparse matrix
        filtered_similarity_matrix = coo_matrix(
            (filtered_data, (filtered_rows, filtered_cols)), 
            shape=similarity_matrix.shape
        )
        
        # Make symmetric: if either (i,j) or (j,i) is selected, keep both
        filtered_csr = filtered_similarity_matrix.tocsr()
        symmetric_matrix = (filtered_csr + filtered_csr.T) / 2
        
        print(f"    {entity_type.capitalize()} adaptive top-K: {symmetric_matrix.nnz} edges, "
              f"avg degree: {symmetric_matrix.nnz / similarity_matrix.shape[0]:.1f}")
        
        return symmetric_matrix
    
    def _adaptive_threshold_filter(self, similarity_matrix, base_threshold, interaction_counts, avg_interaction_count, entity_type):
        """
        Adaptive threshold filtering (our enhancement)
        Higher interaction users get higher thresholds (more selective)
        Lower interaction users get lower thresholds (more inclusive)
        """
        print(f"    Applying adaptive threshold filtering...")
        
        # Convert to dense for threshold operations
        if hasattr(similarity_matrix, 'toarray'):
            similarity_dense = similarity_matrix.toarray()
        else:
            similarity_dense = similarity_matrix
        
        # Create adaptive threshold matrix
        n = similarity_dense.shape[0]
        adaptive_matrix = np.zeros_like(similarity_dense)
        
        for i in range(n):
            # Calculate adaptive threshold based on interaction pattern
            # More interactions → higher threshold (more selective)
            # Fewer interactions → lower threshold (more inclusive)
            interaction_ratio = np.sqrt(interaction_counts[i]) / np.sqrt(avg_interaction_count)
            
            # Adaptive threshold: scale base threshold by interaction ratio
            current_threshold = base_threshold * max(0.5, min(2.0, interaction_ratio))
            current_threshold = max(self.min_threshold, min(current_threshold, self.max_threshold))
            
            # Apply threshold to row
            row = similarity_dense[i]
            adaptive_matrix[i] = np.where(row >= current_threshold, row, 0.0)
        
        # Make symmetric
        adaptive_matrix = (adaptive_matrix + adaptive_matrix.T) / 2
        
        # Convert back to sparse
        from scipy.sparse import csr_matrix
        result_matrix = csr_matrix(adaptive_matrix)
        
        print(f"    {entity_type.capitalize()} adaptive threshold: {result_matrix.nnz} edges, "
              f"avg degree: {result_matrix.nnz / n:.1f}")
        
        return result_matrix
    
    def _create_dysimgcf_similarity_matrices(self):
        """
        Create DySimGCF-style similarity matrices - keep them separate for efficiency
        """
        print("Creating DySimGCF-style similarity matrices...")
        
        # Convert to numpy for similarity computation (DySimGCF approach)
        user_item_matrix = self.adj_tensor.cpu().numpy()
        
        # User-user similarity using adaptive filtering
        print(f"  User-user similarity ({self.u_sim})...")
        if self.u_sim == 'cos':
            user_similarity_matrix = self._adaptive_cosine_similarity(
                user_item_matrix, self.u_threshold, self.u_K, self.self_loop, 'user'
            )
        elif self.u_sim == 'jac':
            user_similarity_matrix = self._adaptive_jaccard_similarity(
                user_item_matrix, self.u_threshold, self.u_K, self.self_loop, 'user'
            )
        else:
            raise ValueError(f"Unsupported user similarity type: {self.u_sim}")
        
        # Item-item similarity using adaptive filtering
        print(f"  Item-item similarity ({self.i_sim})...")
        if self.i_sim == 'cos':
            item_similarity_matrix = self._adaptive_cosine_similarity(
                user_item_matrix.T, self.i_threshold, self.i_K, self.self_loop, 'item'
            )
        elif self.i_sim == 'jac':
            item_similarity_matrix = self._adaptive_jaccard_similarity(
                user_item_matrix.T, self.i_threshold, self.i_K, self.self_loop, 'item'
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
    
    def _initialize_model_enhanced(self):
        """Enhanced initialization: combine direct adjacency with similarity-based approaches"""
        start = time.time()
        
        print(f"Computing enhanced eigendecompositions for filter type: {self.filter}")
        
        # Initialize filters for direct adjacency (like single model)
        self.direct_user_filter = None
        self.direct_item_filter = None
        
        # Initialize filters for similarity-based adjacency
        self.sim_user_filter = None
        self.sim_item_filter = None
        
        # Direct adjacency eigendecomposition (like single model)
        if self.filter in ['u', 'ui']:
            print("Processing direct user similarity...")
            self.direct_user_filter = self._create_direct_filter('user')
            self._memory_cleanup()
        
        if self.filter in ['i', 'ui']:
            print("Processing direct item similarity...")
            self.direct_item_filter = self._create_direct_filter('item')
            self._memory_cleanup()
        
        # Similarity-based eigendecomposition (DySimGCF style)
        if self.filter in ['u', 'ui']:
            print("Processing DySimGCF user similarity...")
            self.sim_user_filter = self._create_similarity_filter('user')
            self._memory_cleanup()
        
        if self.filter in ['i', 'ui']:
            print("Processing DySimGCF item similarity...")
            self.sim_item_filter = self._create_similarity_filter('item')
            self._memory_cleanup()
        
        # Enhanced combination weights (direct + similarity-based)
        if self.filter == 'u':
            # direct + direct_user + sim_user
            self.combination_weights = nn.Parameter(torch.tensor([0.5, 0.3, 0.2]))
        elif self.filter == 'i':
            # direct + direct_item + sim_item
            self.combination_weights = nn.Parameter(torch.tensor([0.5, 0.3, 0.2]))
        else:  # 'ui'
            # direct + direct_item + direct_user + sim_item + sim_user
            self.combination_weights = nn.Parameter(torch.tensor([0.4, 0.2, 0.15, 0.15, 0.1]))
        
        end = time.time()
        print(f'Enhanced initialization time for Universal-SpectralCF ({self.filter}): {end-start:.2f}s')
    
    def _create_direct_filter(self, filter_type):
        """Create filter using direct adjacency (like single model)"""
        print(f"  Computing direct {filter_type} similarity matrix...")
        print(f"  🔍 Debug: norm_adj exists: {hasattr(self, 'norm_adj')}")
        print(f"  🔍 Debug: all buffers: {list(self._buffers.keys())}")
        
        # Check if norm_adj exists, if not create it
        if not hasattr(self, 'norm_adj'):
            print("  ⚠️  norm_adj missing, creating it now...")
            row_sums = self.adj_tensor.sum(dim=1, keepdim=True) + 1e-8
            col_sums = self.adj_tensor.sum(dim=0, keepdim=True) + 1e-8
            norm_adj = self.adj_tensor / torch.sqrt(row_sums) / torch.sqrt(col_sums)
            self.register_buffer('norm_adj', norm_adj)
            del row_sums, col_sums, norm_adj
        
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
        
        sim_np = similarity_matrix.cpu().numpy()
        del similarity_matrix
        self._memory_cleanup()
        
        k = min(self.n_eigen, n_components - 2)
        
        try:
            eigenvals, eigenvecs = eigsh(sp.csr_matrix(sim_np), k=k, which='LM')
            self.register_buffer(f'direct_{filter_type}_eigenvals', 
                               torch.tensor(np.real(eigenvals), dtype=torch.float32))
            self.register_buffer(f'direct_{filter_type}_eigenvecs', 
                               torch.tensor(np.real(eigenvecs), dtype=torch.float32))
            print(f"    Direct {filter_type} eigendecomposition: {k} components, max eigenval = {eigenvals.max():.4f}")
        except Exception as e:
            print(f"    Direct {filter_type} eigendecomposition failed: {e}")
            self.register_buffer(f'direct_{filter_type}_eigenvals', torch.ones(k))
            self.register_buffer(f'direct_{filter_type}_eigenvecs', torch.eye(n_components, k))
        
        del sim_np
        if 'eigenvals' in locals():
            del eigenvals, eigenvecs
        self._memory_cleanup()
        
        return self._create_filter_from_design()
    
    def _create_similarity_filter(self, filter_type):
        """Create filter using DySimGCF similarity matrices"""
        print(f"  Processing DySimGCF {filter_type} similarity matrix...")
        
        # Use the precomputed DySimGCF similarity matrix directly
        if filter_type == 'user':
            sim_np = self.user_sim_adj.cpu().numpy()
            n_components = self.n_users
        else:
            sim_np = self.item_sim_adj.cpu().numpy()
            n_components = self.n_items
        
        k = min(self.n_eigen, n_components - 2)
        
        try:
            eigenvals, eigenvecs = eigsh(sp.csr_matrix(sim_np), k=k, which='LM')
            self.register_buffer(f'sim_{filter_type}_eigenvals', 
                               torch.tensor(np.real(eigenvals), dtype=torch.float32))
            self.register_buffer(f'sim_{filter_type}_eigenvecs', 
                               torch.tensor(np.real(eigenvecs), dtype=torch.float32))
            print(f"    Similarity {filter_type} eigendecomposition: {k} components, max eigenval = {eigenvals.max():.4f}")
        except Exception as e:
            print(f"    Similarity {filter_type} eigendecomposition failed: {e}")
            self.register_buffer(f'sim_{filter_type}_eigenvals', torch.ones(k))
            self.register_buffer(f'sim_{filter_type}_eigenvecs', torch.eye(n_components, k))
        
        del sim_np
        if 'eigenvals' in locals():
            del eigenvals, eigenvecs
        self._memory_cleanup()
        
        return self._create_filter_from_design()
    
    def _get_filter_matrices(self):
        """Compute filter matrices for both direct and similarity-based adjacencies"""
        direct_user_matrix = direct_item_matrix = None
        sim_user_matrix = sim_item_matrix = None
        
        # Direct adjacency filters (like single model)
        if self.filter in ['u', 'ui'] and self.direct_user_filter is not None:
            response = self.direct_user_filter(self.direct_user_eigenvals)
            direct_user_matrix = self.direct_user_eigenvecs @ torch.diag(response) @ self.direct_user_eigenvecs.t()
        
        if self.filter in ['i', 'ui'] and self.direct_item_filter is not None:
            response = self.direct_item_filter(self.direct_item_eigenvals)
            direct_item_matrix = self.direct_item_eigenvecs @ torch.diag(response) @ self.direct_item_eigenvecs.t()
        
        # Similarity-based filters (DySimGCF style)
        if self.filter in ['u', 'ui'] and self.sim_user_filter is not None:
            response = self.sim_user_filter(self.sim_user_eigenvals)
            sim_user_matrix = self.sim_user_eigenvecs @ torch.diag(response) @ self.sim_user_eigenvecs.t()
        
        if self.filter in ['i', 'ui'] and self.sim_item_filter is not None:
            response = self.sim_item_filter(self.sim_item_eigenvals)
            sim_item_matrix = self.sim_item_eigenvecs @ torch.diag(response) @ self.sim_item_eigenvecs.t()
        
        return direct_user_matrix, direct_item_matrix, sim_user_matrix, sim_item_matrix
    
    def forward(self, users):
        """Enhanced forward pass - combines direct and similarity approaches"""
        user_profiles = self.adj_tensor[users]
        direct_scores = user_profiles
        
        # Get both direct and similarity filter matrices
        direct_user_matrix, direct_item_matrix, sim_user_matrix, sim_item_matrix = self._get_filter_matrices()
        
        # Compute scores based on filter type
        if self.filter == 'u':
            # User-based: direct + direct_user + sim_user
            scores = [direct_scores]
            
            if direct_user_matrix is not None:
                direct_user_scores = direct_user_matrix[users] @ self.adj_tensor
                scores.append(direct_user_scores)
            
            if sim_user_matrix is not None:
                sim_user_scores = sim_user_matrix[users] @ self.adj_tensor
                scores.append(sim_user_scores)
            
        elif self.filter == 'i':
            # Item-based: direct + direct_item + sim_item
            scores = [direct_scores]
            
            if direct_item_matrix is not None:
                direct_item_scores = user_profiles @ direct_item_matrix
                scores.append(direct_item_scores)
            
            if sim_item_matrix is not None:
                sim_item_scores = user_profiles @ sim_item_matrix
                scores.append(sim_item_scores)
            
        else:  # 'ui'
            # Combined: direct + direct_item + direct_user + sim_item + sim_user
            scores = [direct_scores]
            
            if direct_item_matrix is not None:
                direct_item_scores = user_profiles @ direct_item_matrix
                scores.append(direct_item_scores)
            
            if direct_user_matrix is not None:
                direct_user_scores = direct_user_matrix[users] @ self.adj_tensor
                scores.append(direct_user_scores)
            
            if sim_item_matrix is not None:
                sim_item_scores = user_profiles @ sim_item_matrix
                scores.append(sim_item_scores)
            
            if sim_user_matrix is not None:
                sim_user_scores = sim_user_matrix[users] @ self.adj_tensor
                scores.append(sim_user_scores)
        
        # Combine with learnable weights
        weights = torch.softmax(self.combination_weights[:len(scores)], dim=0)
        predicted = sum(w * score for w, score in zip(weights, scores))
        
        # Memory cleanup for large datasets
        if self.training and (self.n_users > 10000 or self.n_items > 10000):
            del direct_user_matrix, direct_item_matrix, sim_user_matrix, sim_item_matrix
            self._memory_cleanup()
        
        return predicted
        
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
        if self.direct_user_filter is not None:
            filter_params.extend(self.direct_user_filter.parameters())
        if self.direct_item_filter is not None:
            filter_params.extend(self.direct_item_filter.parameters())
        if self.sim_user_filter is not None:
            filter_params.extend(self.sim_user_filter.parameters())
        if self.sim_item_filter is not None:
            filter_params.extend(self.sim_item_filter.parameters())
        return filter_params
    
    def get_other_parameters(self):
        """Get non-filter parameters"""
        filter_param_ids = {id(p) for p in self.get_filter_parameters()}
        return [p for p in self.parameters() if id(p) not in filter_param_ids]

    def debug_filter_learning(self):
        """Enhanced debug for different filter designs with DySimGCF-style similarities"""
        print(f"\n=== FILTER LEARNING DEBUG (DYSIMGCF-STYLE) - {self.filter_design.upper()} ===")
        if self.adaptive:
            if self.use_topk:
                print(f"Adaptive Top-K: User {self.u_sim} (base-K={self.u_K}), Item {self.i_sim} (base-K={self.i_K})")
            else:
                print(f"Adaptive Threshold: User {self.u_sim} (base={self.u_threshold}), Item {self.i_sim} (base={self.i_threshold})")
        else:
            if self.use_topk:
                print(f"Fixed Top-K: User {self.u_sim} (K={self.u_K}), Item {self.i_sim} (K={self.i_K})")
            else:
                print(f"Fixed Threshold: User {self.u_sim} (≥{self.u_threshold}), Item {self.i_sim} (≥{self.i_threshold})")
        
        with torch.no_grad():
            # Debug direct filters
            if self.filter in ['u', 'ui'] and self.direct_user_filter is not None:
                print(f"\n👤 Direct User Filter ({self.filter_design}):")
                self._debug_single_filter(self.direct_user_filter, "Direct User")
            
            if self.filter in ['i', 'ui'] and self.direct_item_filter is not None:
                print(f"\n🎬 Direct Item Filter ({self.filter_design}):")
                self._debug_single_filter(self.direct_item_filter, "Direct Item")
            
            # Debug similarity filters
            if self.filter in ['u', 'ui'] and self.sim_user_filter is not None:
                print(f"\n👤 Similarity User Filter ({self.filter_design}):")
                self._debug_single_filter(self.sim_user_filter, "Similarity User")
            
            if self.filter in ['i', 'ui'] and self.sim_item_filter is not None:
                print(f"\n🎬 Similarity Item Filter ({self.filter_design}):")
                self._debug_single_filter(self.sim_item_filter, "Similarity Item")
            
            # Combination weights
            weights = torch.softmax(self.combination_weights, dim=0)
            if self.filter == 'u':
                weight_names = ['Direct', 'DirectUser', 'SimUser']
            elif self.filter == 'i':
                weight_names = ['Direct', 'DirectItem', 'SimItem']
            else:  # 'ui'
                weight_names = ['Direct', 'DirectItem', 'DirectUser', 'SimItem', 'SimUser']
            
            print(f"\n🔗 Combination Weights:")
            for i, (name, weight) in enumerate(zip(weight_names[:len(weights)], weights)):
                print(f"  {name:12}: {weight.cpu().item():.4f}")
        
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