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
import gc
import filters as fl

class UniversalSpectralFilter(nn.Module):
    def __init__(self, filter_order=6):
        super().__init__()
        self.filter_order = filter_order

        
        # Initialize coefficients based on a smooth low-pass filter
        #smooth_lowpass = [1.0, -0.5, 0.1, -0.02, 0.004, -0.0008, 0.00015, -0.00003]
        lowpass = fl.get_filter_coefficients('oscillatory_gentle', as_tensor=True) # oscillatory_gentle - 0.398638
        coeffs_data = torch.zeros(filter_order + 1)
        for i, val in enumerate(lowpass[:filter_order + 1]):
            coeffs_data[i] = val

        print(f"Initializing UniversalSpectralFilter with order {filter_order}")
        print(f"  Initial coefficients: {coeffs_data.cpu().numpy()}")
        print(f"  Coefficients shape: {coeffs_data.shape}")
        
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
    def __init__(self, adj_mat, config=None):
        super().__init__()
        
        # Configuration
        self.config = config or {}
        self.device = self.config.get('device', 'cpu')
        self.n_eigen = self.config.get('n_eigen', 50)
        self.filter_order = self.config.get('filter_order', 6)
        self.filter = self.config.get('filter', 'ui')
        
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
        """Setup spectral filters with memory-efficient eigendecompositions"""
        print(f"Computing eigendecompositions for filter type: {self.filter}")
        start = time.time()
        
        # Process filters one at a time to reduce peak memory usage
        self.user_filter = None
        self.item_filter = None
        
        if self.filter in ['u', 'ui']:
            print("Processing user similarity...")
            self.user_filter = self._create_filter_memory_efficient('user')
            self._memory_cleanup()  # Clean up after user processing
        
        if self.filter in ['i', 'ui']:
            print("Processing item similarity...")
            self.item_filter = self._create_filter_memory_efficient('item')
            self._memory_cleanup()  # Clean up after item processing
        
        print(f'Filter setup completed in {time.time() - start:.2f}s')
    
    def _create_filter_memory_efficient(self, filter_type):
        """Create filter with memory-efficient eigendecomposition"""
        print(f"  Computing {filter_type} similarity matrix...")
        
        # Compute similarity matrix with memory management
        with torch.no_grad():  # Disable gradients for initialization
            if filter_type == 'user':
                # Compute user similarity in chunks if needed
                similarity_matrix = self._compute_similarity_chunked(
                    self.norm_adj, self.norm_adj.t(), chunk_size=1000
                )
                n_components = self.n_users
            else:  # item
                # Compute item similarity in chunks if needed  
                similarity_matrix = self._compute_similarity_chunked(
                    self.norm_adj.t(), self.norm_adj, chunk_size=1000
                )
                n_components = self.n_items
        
        # Convert to numpy and compute eigendecomposition
        print(f"  Converting to numpy and computing eigendecomposition...")
        sim_np = similarity_matrix.cpu().numpy()
        
        # Clear similarity matrix from GPU memory immediately
        del similarity_matrix
        self._memory_cleanup()
        
        k = min(self.n_eigen, n_components - 2)
        
        try:
            print(f"  Computing {k} eigenvalues...")
            eigenvals, eigenvecs = eigsh(sp.csr_matrix(sim_np), k=k, which='LM')
            
            # Store eigendecomposition
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
        
        # Clean up numpy arrays
        del sim_np
        if 'eigenvals' in locals():
            del eigenvals, eigenvecs
        self._memory_cleanup()
        
        return UniversalSpectralFilter(self.filter_order)
    
    def _compute_similarity_chunked(self, A, B, chunk_size=1000):
        """Compute A @ B in chunks to save memory"""
        # For very large matrices, compute in chunks
        if A.shape[0] <= chunk_size:
            return A @ B
        
        print(f"    Using chunked computation (chunk_size={chunk_size})...")
        result_chunks = []
        
        for i in range(0, A.shape[0], chunk_size):
            end_idx = min(i + chunk_size, A.shape[0])
            chunk_result = A[i:end_idx] @ B
            result_chunks.append(chunk_result)
            
            # Clean up intermediate results
            if i > 0:  # Keep some chunks in memory for efficiency
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
        
        # Get filter matrices (these are computed on-demand)
        user_filter_matrix, item_filter_matrix = self._get_filter_matrices()
        
        # Compute filtered scores
        scores = [user_profiles]  # Direct scores
        
        if self.filter in ['i', 'ui'] and item_filter_matrix is not None:
            scores.append(user_profiles @ item_filter_matrix)
        
        if self.filter in ['u', 'ui'] and user_filter_matrix is not None:
            user_filtered = user_filter_matrix[users] @ self.adj_tensor
            scores.append(user_filtered)
        
        # Combine scores with learned weights
        weights = torch.softmax(self.combination_weights, dim=0)
        predicted = sum(w * score for w, score in zip(weights, scores))
        
        # Clean up intermediate matrices if they're large
        if self.training and (self.n_users > 10000 or self.n_items > 10000):
            del user_filter_matrix, item_filter_matrix
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
        """Debug what the filters are learning and identify filter patterns"""
        print("\n=== FILTER LEARNING DEBUG ===")
        
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
                # Compare with same number of coefficients
                pattern_truncated = pattern_coeffs[:len(coeffs)]
                
                # Calculate correlation coefficient
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
            
            # Analyze coefficient pattern
            c0, c1 = coeffs[0], coeffs[1]
            
            # Check for different behaviors
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
            # Analyze user filter
            if self.filter in ['u', 'ui'] and self.user_filter is not None:
                user_match, user_sim, user_type = analyze_filter_pattern(
                    self.user_filter.coeffs, "User"
                )
                user_response = self.user_filter(self.user_eigenvals)
                print(f"  Filter response range: [{user_response.min():.4f}, {user_response.max():.4f}]")
            
            # Analyze item filter
            if self.filter in ['i', 'ui'] and self.item_filter is not None:
                item_match, item_sim, item_type = analyze_filter_pattern(
                    self.item_filter.coeffs, "Item"
                )
                item_response = self.item_filter(self.item_eigenvals)
                print(f"  Filter response range: [{item_response.min():.4f}, {item_response.max():.4f}]")
            
            # Combination weights analysis
            weights = torch.softmax(self.combination_weights, dim=0)
            print(f"\n🔗 Combination Weights Analysis:")
            print(f"  Raw weights: {weights.cpu().numpy()}")
            
            if self.filter == 'ui':
                direct, item, user = weights.cpu().numpy()
                print(f"  📈 Component Importance:")
                print(f"     └─ Direct CF: {direct:.3f} ({'🔥 Dominant' if direct > 0.5 else '🔸 Moderate' if direct > 0.3 else '🔹 Minor'})")
                print(f"     └─ Item filtering: {item:.3f} ({'🔥 Dominant' if item > 0.5 else '🔸 Moderate' if item > 0.3 else '🔹 Minor'})")
                print(f"     └─ User filtering: {user:.3f} ({'🔥 Dominant' if user > 0.5 else '🔸 Moderate' if user > 0.3 else '🔹 Minor'})")
            elif self.filter == 'u':
                direct, user = weights.cpu().numpy()
                print(f"     └─ Direct CF: {direct:.3f}")
                print(f"     └─ User filtering: {user:.3f}")
            elif self.filter == 'i':
                direct, item = weights.cpu().numpy()
                print(f"     └─ Direct CF: {direct:.3f}")
                print(f"     └─ Item filtering: {item:.3f}")
            
            # Overall model interpretation
            print(f"\n🎯 Overall Model Interpretation:")
            if self.filter == 'ui':
                if hasattr(locals(), 'user_match') and hasattr(locals(), 'item_match'):
                    print(f"  └─ User-side learned: {user_type} ({user_match}-like)")
                    print(f"  └─ Item-side learned: {item_type} ({item_match}-like)")
                    
                    # Suggest what this means
                    if user_type.startswith('low-pass') and item_type.startswith('low-pass'):
                        print(f"  🔍 Model focuses on global patterns (popular items, broad preferences)")
                    elif 'high-pass' in user_type or 'high-pass' in item_type:
                        print(f"  🔍 Model emphasizes niche patterns (specific preferences)")
                    else:
                        print(f"  🔍 Model learned balanced filtering strategy")
            
        print("=== END DEBUG ===\n")