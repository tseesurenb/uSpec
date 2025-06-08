'''
Created on June 7, 2025
PyTorch Implementation of Laplacian-based Universal Spectral Collaborative Filtering

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
        
        self.config = config or {}
        self.device = self.config.get('device', 'cpu')
        self.n_eigen = self.config.get('n_eigen', 50)
        self.filter_order = self.config.get('filter_order', 6)
        self.filter = self.config.get('filter', 'ui')
        
        # Laplacian configuration
        self.laplacian_type = self.config.get('laplacian_type', 'normalized_sym')
        self.similarity_threshold = self.config.get('similarity_threshold', 0.01)
        self.filter_design = self.config.get('filter_design', 'enhanced_basis')
        self.init_filter = self.config.get('init_filter', 'smooth')
        
        # Convert adjacency matrix
        adj_dense = adj_mat.toarray() if sp.issparse(adj_mat) else adj_mat
        self.register_buffer('adj_tensor', torch.tensor(adj_dense, dtype=torch.float32))
        self.n_users, self.n_items = self.adj_tensor.shape
        
        # Setup filters and weights
        self._setup_laplacian_filters()
        self._setup_combination_weights()
        
        del adj_dense
        self._memory_cleanup()
    
    def _memory_cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _setup_laplacian_filters(self):
        self.user_filter = None
        self.item_filter = None
        
        if self.filter in ['u', 'ui']:
            self.user_filter = self._create_laplacian_filter('user')
            self._memory_cleanup()
        
        if self.filter in ['i', 'ui']:
            self.item_filter = self._create_laplacian_filter('item')
            self._memory_cleanup()
    
    def _create_laplacian_filter(self, filter_type):
        # Compute similarity matrix
        with torch.no_grad():
            norm_adj = self._normalize_adjacency(self.adj_tensor)
            if filter_type == 'user':
                similarity_matrix = self._compute_similarity_chunked(
                    norm_adj, norm_adj.t(), chunk_size=1000
                )
                n_components = self.n_users
            else:
                similarity_matrix = self._compute_similarity_chunked(
                    norm_adj.t(), norm_adj, chunk_size=1000
                )
                n_components = self.n_items
        
        # Convert similarity to adjacency
        adjacency_matrix = self._threshold_similarity(similarity_matrix)
        
        # Construct Laplacian matrix
        laplacian_matrix = self._construct_laplacian(adjacency_matrix)
        
        # Compute eigendecomposition
        eigenvals, eigenvecs = self._compute_laplacian_eigen(laplacian_matrix, n_components)
        
        # Store eigendecomposition
        self.register_buffer(f'{filter_type}_eigenvals', eigenvals)
        self.register_buffer(f'{filter_type}_eigenvecs', eigenvecs)
        
        # Create spectral filter
        return self._create_spectral_filter()
    
    def _normalize_adjacency(self, adj_tensor):
        row_sums = adj_tensor.sum(dim=1, keepdim=True) + 1e-8
        col_sums = adj_tensor.sum(dim=0, keepdim=True) + 1e-8
        return adj_tensor / torch.sqrt(row_sums) / torch.sqrt(col_sums)
    
    def _threshold_similarity(self, similarity_matrix):
        adjacency = (similarity_matrix > self.similarity_threshold).float()
        adjacency = (adjacency + adjacency.t()) / 2.0
        return (adjacency > 0.5).float()
    
    def _construct_laplacian(self, adjacency_matrix):
        degrees = adjacency_matrix.sum(dim=1)
        degrees = torch.clamp(degrees, min=1e-8)
        
        if self.laplacian_type == 'unnormalized':
            D = torch.diag(degrees)
            L = D - adjacency_matrix
        elif self.laplacian_type == 'normalized_sym':
            D_sqrt_inv = torch.diag(1.0 / torch.sqrt(degrees))
            normalized_adj = D_sqrt_inv @ adjacency_matrix @ D_sqrt_inv
            L = torch.eye(adjacency_matrix.shape[0], device=adjacency_matrix.device) - normalized_adj
        elif self.laplacian_type == 'normalized_rw':
            D_inv = torch.diag(1.0 / degrees)
            normalized_adj = D_inv @ adjacency_matrix
            L = torch.eye(adjacency_matrix.shape[0], device=adjacency_matrix.device) - normalized_adj
        else:
            raise ValueError(f"Unknown Laplacian type: {self.laplacian_type}")
        
        return L
    
    def _compute_laplacian_eigen(self, laplacian_matrix, n_components):
        lap_np = laplacian_matrix.cpu().numpy()
        del laplacian_matrix
        self._memory_cleanup()
        
        k = min(self.n_eigen, n_components - 2)
        
        try:
            eigenvals, eigenvecs = eigsh(sp.csr_matrix(lap_np), k=k, which='SM', sigma=0.1)
            eigenvals = np.maximum(eigenvals, 0.0)
            eigenvals_tensor = torch.tensor(np.real(eigenvals), dtype=torch.float32)
            eigenvecs_tensor = torch.tensor(np.real(eigenvecs), dtype=torch.float32)
        except Exception:
            eigenvals_tensor = torch.linspace(0, 1, min(self.n_eigen, n_components))
            eigenvecs_tensor = torch.eye(n_components, min(self.n_eigen, n_components))
        
        del lap_np
        if 'eigenvals' in locals():
            del eigenvals, eigenvecs
        self._memory_cleanup()
        
        return eigenvals_tensor, eigenvecs_tensor
    
    def _create_spectral_filter(self):
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
        elif self.filter_design == 'deep':
            return fl.DeepSpectralFilter(self.filter_order, self.init_filter)
        elif self.filter_design == 'multiscale':
            return fl.MultiScaleSpectralFilter(self.filter_order, self.init_filter)
        elif self.filter_design == 'ensemble':
            return fl.EnsembleSpectralFilter(self.filter_order, self.init_filter)
        else:
            raise ValueError(f"Unknown filter design: {self.filter_design}")
    
    def _compute_similarity_chunked(self, A, B, chunk_size=1000):
        if A.shape[0] <= chunk_size:
            return A @ B
        
        result_chunks = []
        for i in range(0, A.shape[0], chunk_size):
            end_idx = min(i + chunk_size, A.shape[0])
            chunk_result = A[i:end_idx] @ B
            result_chunks.append(chunk_result)
            if i > 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return torch.cat(result_chunks, dim=0)
    
    def _setup_combination_weights(self):
        init_weights = {'u': [0.5, 0.5], 'i': [0.5, 0.5], 'ui': [0.5, 0.3, 0.2]}
        self.combination_weights = nn.Parameter(torch.tensor(init_weights[self.filter]))
    
    def _get_filter_matrices(self):
        user_matrix = item_matrix = None
        
        if self.user_filter is not None:
            response = self.user_filter(self.user_eigenvals)
            user_matrix = self.user_eigenvecs @ torch.diag(response) @ self.user_eigenvecs.t()
        
        if self.item_filter is not None:
            response = self.item_filter(self.item_eigenvals)
            item_matrix = self.item_eigenvecs @ torch.diag(response) @ self.item_eigenvecs.t()
        
        return user_matrix, item_matrix
    
    def forward(self, users):
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
        self.eval()
        with torch.no_grad():
            if isinstance(batch_users, np.ndarray):
                batch_users = torch.LongTensor(batch_users)
            
            result = self.forward(batch_users).cpu().numpy()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return result
    
    def get_filter_parameters(self):
        filter_params = []
        if self.user_filter is not None:
            filter_params.extend(self.user_filter.parameters())
        if self.item_filter is not None:
            filter_params.extend(self.item_filter.parameters())
        return filter_params
    
    def get_other_parameters(self):
        filter_param_ids = {id(p) for p in self.get_filter_parameters()}
        return [p for p in self.parameters() if id(p) not in filter_param_ids]
    
    def debug_filter_learning(self):
        print(f"\n=== LAPLACIAN SPECTRAL FILTER DEBUG ({self.filter_design.upper()}) ===")
        
        with torch.no_grad():
            if self.filter in ['u', 'ui'] and self.user_filter is not None:
                print(f"\n👤 User Laplacian Filter:")
                eigenvals = self.user_eigenvals.cpu().numpy()
                print(f"  Eigenvalue range: [{eigenvals.min():.6f}, {eigenvals.max():.6f}]")
                zero_eigenvals = (eigenvals < 1e-6).sum()
                print(f"  Zero eigenvalues: {zero_eigenvals}")
                
                response = self.user_filter(self.user_eigenvals).cpu().numpy()
                print(f"  Filter response range: [{response.min():.6f}, {response.max():.6f}]")
                
                self._debug_single_filter(self.user_filter, "User")
            
            if self.filter in ['i', 'ui'] and self.item_filter is not None:
                print(f"\n🎬 Item Laplacian Filter:")
                eigenvals = self.item_eigenvals.cpu().numpy()
                print(f"  Eigenvalue range: [{eigenvals.min():.6f}, {eigenvals.max():.6f}]")
                zero_eigenvals = (eigenvals < 1e-6).sum()
                print(f"  Zero eigenvalues: {zero_eigenvals}")
                
                response = self.item_filter(self.item_eigenvals).cpu().numpy()
                print(f"  Filter response range: [{response.min():.6f}, {response.max():.6f}]")
                
                self._debug_single_filter(self.item_filter, "Item")
            
            weights = torch.softmax(self.combination_weights, dim=0)
            print(f"\n🔗 Combination Weights: {weights.cpu().numpy()}")
        
        print("=== END DEBUG ===\n")
    
    def _debug_single_filter(self, filter_obj, filter_name):
        if isinstance(filter_obj, fl.UniversalSpectralFilter):
            init_coeffs = filter_obj.init_coeffs.cpu().numpy()
            current_coeffs = filter_obj.coeffs.cpu().numpy()
            print(f"  Initial filter: {filter_obj.init_filter_name}")
            change = current_coeffs - init_coeffs
            print(f"  Max coefficient change: {np.abs(change).max():.6f}")
            
        elif isinstance(filter_obj, fl.EnhancedSpectralBasisFilter):
            mixing_analysis = filter_obj.get_mixing_analysis()
            print(f"  Top filter combinations:")
            for name, weight in list(mixing_analysis.items())[:3]:
                print(f"    {name}: {weight:.4f}")
    
    def get_memory_usage(self):
        stats = {
            'parameters': sum(p.numel() * p.element_size() for p in self.parameters()) / 1024**2,
            'buffers': sum(b.numel() * b.element_size() for b in self.buffers()) / 1024**2,
        }
        
        if torch.cuda.is_available():
            stats['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**2
            stats['gpu_cached'] = torch.cuda.memory_reserved() / 1024**2
        
        return {k: f"{v:.2f} MB" for k, v in stats.items()}
    
    def get_parameter_count(self):
        total_params = sum(p.numel() for p in self.parameters())
        filter_params = sum(p.numel() for p in self.get_filter_parameters())
        other_params = total_params - filter_params
        
        return {
            'total': total_params,
            'filter': filter_params,
            'combination_weights': self.combination_weights.numel(),
            'other': other_params
        }