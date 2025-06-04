'''
Created on June 3, 2025
PyTorch Implementation of uSpec: Universal Spectral Collaborative Filtering

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import time
import world
import gc

class UniversalSpectralFilter(nn.Module):
    def __init__(self, filter_order=3):
        super().__init__()
        self.filter_order = filter_order

        # Initialize coefficients based on a smooth low-pass filter
        smooth_lowpass = [1.0, -0.5, 0.1, -0.02, 0.004, -0.0008, 0.00015, -0.00003]
        coeffs_data = torch.zeros(filter_order + 1)
        for i, val in enumerate(smooth_lowpass[:filter_order + 1]):
            coeffs_data[i] = val
        
        self.coeffs = nn.Parameter(coeffs_data)
    
    def forward(self, eigenvalues):
        """Apply learnable spectral filter using Chebyshev polynomials"""
        # Normalize eigenvalues to [-1, 1]
        max_eigenval = torch.max(eigenvalues) + 1e-8
        x = 2 * (eigenvalues / max_eigenval) - 1
        
        # Compute Chebyshev polynomial response
        result = self.coeffs[0] * torch.ones_like(x)
        
        if len(self.coeffs) > 1:
            T_prev, T_curr = torch.ones_like(x), x
            result += self.coeffs[1] * T_curr
            
            for i in range(2, len(self.coeffs)):
                T_next = 2 * x * T_curr - T_prev
                result += self.coeffs[i] * T_next
                T_prev, T_curr = T_curr, T_next
        
        return torch.exp(-torch.abs(result)) + 1e-6

class UniversalSpectralCF(nn.Module):
    """
    Universal Spectral CF with Positive and Negative Similarities
    Memory-optimized with enhanced debugging
    """
    def __init__(self, adj_mat, config=None):
        super().__init__()
        
        # Configuration
        self.config = config or {}
        self.device = self.config.get('device', 'cpu')
        self.n_eigen = self.config.get('n_eigen', 50)
        self.filter_order = self.config.get('filter_order', 3)
        self.filter = self.config.get('filter', 'ui')
        
        # Convert and register adjacency matrix
        adj_dense = adj_mat.toarray() if sp.issparse(adj_mat) else adj_mat
        self.register_buffer('adj_tensor', torch.tensor(adj_dense, dtype=torch.float32))
        self.n_users, self.n_items = self.adj_tensor.shape
        
        # Compute and register normalized adjacency for positive similarities
        row_sums = self.adj_tensor.sum(dim=1, keepdim=True) + 1e-8
        col_sums = self.adj_tensor.sum(dim=0, keepdim=True) + 1e-8
        norm_adj = self.adj_tensor / torch.sqrt(row_sums) / torch.sqrt(col_sums)
        self.register_buffer('norm_adj', norm_adj)
        
        # Create and register normalized adjacency for negative similarities
        binary_adj = (self.adj_tensor > 0).float()
        complement_adj = 1 - binary_adj
        neg_row_sums = complement_adj.sum(dim=1, keepdim=True) + 1e-8
        neg_col_sums = complement_adj.sum(dim=0, keepdim=True) + 1e-8
        neg_norm_adj = complement_adj / torch.sqrt(neg_row_sums) / torch.sqrt(neg_col_sums)
        self.register_buffer('neg_norm_adj', neg_norm_adj)
        
        # Clean up intermediate variables
        del adj_dense, row_sums, col_sums, binary_adj, complement_adj, neg_row_sums, neg_col_sums
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
        """Setup spectral filters for positive and negative similarities"""
        print(f"Computing positive and negative eigendecompositions for filter type: {self.filter}")
        start = time.time()
        
        # Initialize all filters to None
        self.user_pos_filter = None
        self.user_neg_filter = None
        self.item_pos_filter = None
        self.item_neg_filter = None
        
        # Process user filters
        if self.filter in ['u', 'ui']:
            print("Processing user positive similarity...")
            self.user_pos_filter = self._create_filter_memory_efficient('user', 'pos')
            self._memory_cleanup()
            
            print("Processing user negative similarity...")
            self.user_neg_filter = self._create_filter_memory_efficient('user', 'neg')
            self._memory_cleanup()
        
        # Process item filters
        if self.filter in ['i', 'ui']:
            print("Processing item positive similarity...")
            self.item_pos_filter = self._create_filter_memory_efficient('item', 'pos')
            self._memory_cleanup()
            
            print("Processing item negative similarity...")
            self.item_neg_filter = self._create_filter_memory_efficient('item', 'neg')
            self._memory_cleanup()
        
        print(f'Filter setup completed in {time.time() - start:.2f}s')
    
    def _create_filter_memory_efficient(self, filter_type, similarity_type):
        """Create filter with memory-efficient eigendecomposition"""
        print(f"  Computing {filter_type} {similarity_type} similarity matrix...")
        
        # Select appropriate normalized adjacency matrix
        norm_adj_matrix = self.norm_adj if similarity_type == 'pos' else self.neg_norm_adj
        
        # Compute similarity matrix with memory management
        with torch.no_grad():
            if filter_type == 'user':
                similarity_matrix = self._compute_similarity_chunked(
                    norm_adj_matrix, norm_adj_matrix.t(), chunk_size=1000
                )
                n_components = self.n_users
            else:  # item
                similarity_matrix = self._compute_similarity_chunked(
                    norm_adj_matrix.t(), norm_adj_matrix, chunk_size=1000
                )
                n_components = self.n_items
        
        # Convert to numpy and compute eigendecomposition
        print(f"  Converting to numpy and computing eigendecomposition...")
        sim_np = similarity_matrix.cpu().numpy()
        
        # Clear similarity matrix from memory immediately
        del similarity_matrix
        self._memory_cleanup()
        
        k = min(self.n_eigen, n_components - 2)
        
        try:
            print(f"  Computing {k} eigenvalues...")
            eigenvals, eigenvecs = eigsh(sp.csr_matrix(sim_np), k=k, which='LM')
            
            # Store eigendecomposition with descriptive names
            buffer_name_vals = f'{filter_type}_{similarity_type}_eigenvals'
            buffer_name_vecs = f'{filter_type}_{similarity_type}_eigenvecs'
            
            self.register_buffer(buffer_name_vals, 
                               torch.tensor(np.real(eigenvals), dtype=torch.float32))
            self.register_buffer(buffer_name_vecs, 
                               torch.tensor(np.real(eigenvecs), dtype=torch.float32))
            
            print(f"  {filter_type.capitalize()} {similarity_type} eigendecomposition: {k} components")
            
        except Exception as e:
            print(f"  {filter_type.capitalize()} {similarity_type} eigendecomposition failed: {e}")
            print(f"  Using fallback identity matrices...")
            
            buffer_name_vals = f'{filter_type}_{similarity_type}_eigenvals'
            buffer_name_vecs = f'{filter_type}_{similarity_type}_eigenvecs'
            
            self.register_buffer(buffer_name_vals, 
                               torch.ones(min(self.n_eigen, n_components)))
            self.register_buffer(buffer_name_vecs, 
                               torch.eye(n_components, min(self.n_eigen, n_components)))
        
        # Clean up numpy arrays
        del sim_np
        if 'eigenvals' in locals():
            del eigenvals, eigenvecs
        self._memory_cleanup()
        
        return UniversalSpectralFilter(self.filter_order)
    
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
            
            # Clean up intermediate results
            if i > 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return torch.cat(result_chunks, dim=0)
    
    def _setup_combination_weights(self):
        """Setup learnable combination weights for positive and negative similarities"""
        init_weights = {
            'u': [0.4, 0.3, 0.3],        # [direct, user_pos, user_neg]
            'i': [0.4, 0.3, 0.3],        # [direct, item_pos, item_neg]
            'ui': [0.3, 0.2, 0.2, 0.15, 0.15]  # [direct, item_pos, item_neg, user_pos, user_neg]
        }
        self.combination_weights = nn.Parameter(torch.tensor(init_weights[self.filter]))
    
    def _get_filter_matrices(self):
        """Compute filter matrices for both positive and negative similarities"""
        matrices = {}
        
        if self.filter in ['u', 'ui']:
            if self.user_pos_filter is not None:
                response = self.user_pos_filter(self.user_pos_eigenvals)
                matrices['user_pos'] = self.user_pos_eigenvecs @ torch.diag(response) @ self.user_pos_eigenvecs.t()
            
            if self.user_neg_filter is not None:
                response = self.user_neg_filter(self.user_neg_eigenvals)
                matrices['user_neg'] = self.user_neg_eigenvecs @ torch.diag(response) @ self.user_neg_eigenvecs.t()
        
        if self.filter in ['i', 'ui']:
            if self.item_pos_filter is not None:
                response = self.item_pos_filter(self.item_pos_eigenvals)
                matrices['item_pos'] = self.item_pos_eigenvecs @ torch.diag(response) @ self.item_pos_eigenvecs.t()
            
            if self.item_neg_filter is not None:
                response = self.item_neg_filter(self.item_neg_eigenvals)
                matrices['item_neg'] = self.item_neg_eigenvecs @ torch.diag(response) @ self.item_neg_eigenvecs.t()
        
        return matrices
    
    def forward(self, users):
        """Clean forward pass - ONLY returns predictions"""
        user_profiles = self.adj_tensor[users]
        filter_matrices = self._get_filter_matrices()
        
        # Compute filtered scores based on filter type
        scores = [user_profiles]  # Direct scores always included
        
        if self.filter == 'u':
            # User-based filtering: direct + user_pos + user_neg
            if 'user_pos' in filter_matrices:
                user_pos_scores = filter_matrices['user_pos'][users] @ self.adj_tensor
                scores.append(user_pos_scores)
            
            if 'user_neg' in filter_matrices:
                user_neg_scores = filter_matrices['user_neg'][users] @ self.adj_tensor
                scores.append(user_neg_scores)
                
        elif self.filter == 'i':
            # Item-based filtering: direct + item_pos + item_neg
            if 'item_pos' in filter_matrices:
                scores.append(user_profiles @ filter_matrices['item_pos'])
            
            if 'item_neg' in filter_matrices:
                scores.append(user_profiles @ filter_matrices['item_neg'])
                
        else:  # 'ui'
            # Combined filtering: direct + item_pos + item_neg + user_pos + user_neg
            if 'item_pos' in filter_matrices:
                scores.append(user_profiles @ filter_matrices['item_pos'])
            
            if 'item_neg' in filter_matrices:
                scores.append(user_profiles @ filter_matrices['item_neg'])
            
            if 'user_pos' in filter_matrices:
                user_pos_scores = filter_matrices['user_pos'][users] @ self.adj_tensor
                scores.append(user_pos_scores)
            
            if 'user_neg' in filter_matrices:
                user_neg_scores = filter_matrices['user_neg'][users] @ self.adj_tensor
                scores.append(user_neg_scores)
        
        # Combine scores with learned weights
        weights = torch.softmax(self.combination_weights, dim=0)
        predicted = sum(w * score for w, score in zip(weights, scores))
        
        # Clean up intermediate matrices if they're large
        if self.training and (self.n_users > 10000 or self.n_items > 10000):
            del filter_matrices
            self._memory_cleanup()
        
        return predicted  # ALWAYS return predictions only!
    
    def getUsersRating(self, batch_users):
        """Memory-efficient evaluation interface"""
        self.eval()
        with torch.no_grad():
            if isinstance(batch_users, np.ndarray):
                batch_users = torch.LongTensor(batch_users)
            
            result = self.forward(batch_users).cpu().numpy()
            
            # Clean up GPU memory after evaluation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return result
    
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

    def debug_filter_learning(self):
        """Debug what the filters are learning with enhanced pattern recognition"""
        print("\n=== FILTER LEARNING DEBUG (POS/NEG) ===")
        
        # Known filter patterns for comparison
        filter_patterns = {
            'butterworth': [1.0, -0.6, 0.2, -0.05, 0.01, -0.002, 0.0003, -0.00005],
            'chebyshev': [1.0, -0.4, 0.1, -0.01, 0.001, -0.0001, 0.00001, -0.000001],
            'smooth': [1.0, -0.5, 0.1, -0.02, 0.004, -0.0008, 0.00015, -0.00003],
            'bessel': [1.0, -0.3, 0.06, -0.008, 0.0008, -0.00006, 0.000004, -0.0000002],
            'gaussian': [1.0, -0.7, 0.15, -0.03, 0.005, -0.0007, 0.00008, -0.000008],
            'conservative': [1.0, -0.2, 0.03, -0.002, 0.0001, -0.000005, 0.0000002, -0.00000001],
            'aggressive': [1.0, -0.8, 0.3, -0.08, 0.015, -0.002, 0.0002, -0.00002]
        }
        
        def analyze_filter_pattern(coeffs_tensor, filter_name):
            """Analyze learned coefficients and find closest pattern"""
            coeffs = coeffs_tensor.cpu().numpy()
            print(f"\n{filter_name} Filter Analysis:")
            print(f"  Learned coefficients: {coeffs}")
            
            # Find closest pattern
            best_match = None
            best_similarity = -1
            
            for pattern_name, pattern_coeffs in filter_patterns.items():
                pattern_truncated = pattern_coeffs[:len(coeffs)]
                
                if len(coeffs) > 1 and len(pattern_truncated) > 1:
                    correlation = np.corrcoef(coeffs, pattern_truncated)[0, 1]
                    if not np.isnan(correlation) and correlation > best_similarity:
                        best_similarity = correlation
                        best_match = pattern_name
            
            # Determine filter characteristics
            filter_type = classify_filter_behavior(coeffs)
            
            print(f"  📊 Filter Characteristics:")
            print(f"     └─ Type: {filter_type}")
            print(f"     └─ Closest pattern: {best_match} (similarity: {best_similarity:.3f})")
            
            # Pattern interpretation
            if best_similarity > 0.9:
                print(f"     └─ 🎯 Strong match to {best_match} filter!")
            elif best_similarity > 0.7:
                print(f"     └─ ✅ Good match to {best_match}-like behavior")
            elif best_similarity > 0.5:
                print(f"     └─ 🔄 Moderate similarity to {best_match}")
            else:
                print(f"     └─ 🆕 Learned unique pattern (not matching standard filters)")
            
            return best_match, best_similarity, filter_type
        
        def classify_filter_behavior(coeffs):
            """Classify the learned filter behavior"""
            if len(coeffs) < 2:
                return "constant"
            
            c0, c1 = coeffs[0], coeffs[1]
            
            if abs(c0) > 0.8 and c1 < -0.3:
                if len(coeffs) > 2 and coeffs[2] > 0:
                    return "low-pass (strong)"
                else:
                    return "low-pass (moderate)"
            elif abs(c0) < 0.3 and c1 > 0.3:
                return "high-pass"
            elif c0 > 0.5 and abs(c1) < 0.3:
                return "conservative low-pass"
            elif len(coeffs) > 2 and abs(coeffs[2]) > 0.1:
                return "band-pass/complex"
            else:
                return "custom/mixed"
        
        with torch.no_grad():
            # Analyze user filters
            if self.filter in ['u', 'ui']:
                if self.user_pos_filter is not None:
                    user_pos_match, user_pos_sim, user_pos_type = analyze_filter_pattern(
                        self.user_pos_filter.coeffs, "User Positive"
                    )
                    user_pos_response = self.user_pos_filter(self.user_pos_eigenvals)
                    print(f"  Filter response range: [{user_pos_response.min():.4f}, {user_pos_response.max():.4f}]")
                
                if self.user_neg_filter is not None:
                    user_neg_match, user_neg_sim, user_neg_type = analyze_filter_pattern(
                        self.user_neg_filter.coeffs, "User Negative"
                    )
                    user_neg_response = self.user_neg_filter(self.user_neg_eigenvals)
                    print(f"  Filter response range: [{user_neg_response.min():.4f}, {user_neg_response.max():.4f}]")
            
            # Analyze item filters
            if self.filter in ['i', 'ui']:
                if self.item_pos_filter is not None:
                    item_pos_match, item_pos_sim, item_pos_type = analyze_filter_pattern(
                        self.item_pos_filter.coeffs, "Item Positive"
                    )
                    item_pos_response = self.item_pos_filter(self.item_pos_eigenvals)
                    print(f"  Filter response range: [{item_pos_response.min():.4f}, {item_pos_response.max():.4f}]")
                
                if self.item_neg_filter is not None:
                    item_neg_match, item_neg_sim, item_neg_type = analyze_filter_pattern(
                        self.item_neg_filter.coeffs, "Item Negative"
                    )
                    item_neg_response = self.item_neg_filter(self.item_neg_eigenvals)
                    print(f"  Filter response range: [{item_neg_response.min():.4f}, {item_neg_response.max():.4f}]")
            
            # Combination weights analysis
            weights = torch.softmax(self.combination_weights, dim=0)
            print(f"\n🔗 Combination Weights Analysis:")
            print(f"  Raw weights: {weights.cpu().numpy()}")
            
            weight_labels = {
                'u': ['Direct CF', 'User Positive', 'User Negative'],
                'i': ['Direct CF', 'Item Positive', 'Item Negative'],
                'ui': ['Direct CF', 'Item Positive', 'Item Negative', 'User Positive', 'User Negative']
            }
            
            labels = weight_labels[self.filter]
            print(f"  📈 Component Importance:")
            for i, (label, weight) in enumerate(zip(labels, weights.cpu().numpy())):
                importance = '🔥 Dominant' if weight > 0.4 else '🔸 Moderate' if weight > 0.2 else '🔹 Minor'
                print(f"     └─ {label}: {weight:.3f} ({importance})")
            
            # Overall model interpretation
            print(f"\n🎯 Overall Model Interpretation:")
            print(f"  └─ Model Type: Positive/Negative Similarity Filtering")
            
            if self.filter == 'ui' and hasattr(locals(), 'user_pos_type'):
                print(f"  └─ User positive learned: {user_pos_type}")
                print(f"  └─ User negative learned: {user_neg_type}")
                print(f"  └─ Item positive learned: {item_pos_type}")
                print(f"  └─ Item negative learned: {item_neg_type}")
                
                # Advanced interpretation
                pos_emphasis = weights[1] + weights[3]  # item_pos + user_pos
                neg_emphasis = weights[2] + weights[4]  # item_neg + user_neg
                
                if pos_emphasis > neg_emphasis * 1.5:
                    print(f"  🔍 Model emphasizes positive similarities (collaborative patterns)")
                elif neg_emphasis > pos_emphasis * 1.5:
                    print(f"  🔍 Model emphasizes negative similarities (diverse patterns)")
                else:
                    print(f"  🔍 Model balances positive and negative similarities")
        
        print("=== END DEBUG ===\n")