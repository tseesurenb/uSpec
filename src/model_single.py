'''
Created on June 3, 2025
Pytorch Implementation of uSpec in
Batsuuri. Tse et al. uSpec: Universal Spectral Collaborative Filtering

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import time
import world

class UniversalSpectralFilter(nn.Module):
    def __init__(self, filter_order=3):
        super().__init__()
        self.filter_order = filter_order
        
        # FIXED: Better initialization to match old model behavior
        self.coeffs = nn.Parameter(torch.zeros(filter_order + 1))
        
        # Initialize like the old model's fixed filters
        with torch.no_grad():
            if filter_order >= 0:
                self.coeffs.data[0] = 1.0      # Base response
            if filter_order >= 1:
                self.coeffs.data[1] = -0.5     # First order damping
            if filter_order >= 2:
                self.coeffs.data[2] = 0.1      # Second order correction
            if filter_order >= 3:
                self.coeffs.data[3] = -0.02    # Higher order terms
    
    def forward(self, eigenvalues):
        """Apply learnable spectral filter using Chebyshev polynomials"""
        # Normalize eigenvalues to [-1, 1] for Chebyshev polynomials
        max_eigenval = torch.max(eigenvalues) + 1e-8
        eigenvals_norm = 2 * (eigenvalues / max_eigenval) - 1
        
        # Compute Chebyshev polynomial response
        filter_response = torch.zeros_like(eigenvals_norm)
        
        # T_0(x) = 1
        if len(self.coeffs) > 0:
            filter_response += self.coeffs[0] * torch.ones_like(eigenvals_norm)
        
        if len(self.coeffs) > 1:
            # T_1(x) = x
            T_prev = torch.ones_like(eigenvals_norm)
            T_curr = eigenvals_norm
            filter_response += self.coeffs[1] * T_curr
            
            # T_n(x) = 2x*T_{n-1}(x) - T_{n-2}(x)
            for i in range(2, len(self.coeffs)):
                T_next = 2 * eigenvals_norm * T_curr - T_prev
                filter_response += self.coeffs[i] * T_next
                T_prev, T_curr = T_curr, T_next
        
        # FIXED: Use exponential-like activation to match old model
        # The old model used exp(-eigenvals * 2.0), so we mimic this behavior
        filter_response = torch.exp(-torch.abs(filter_response)) + 1e-6
        
        return filter_response

class UniversalSpectralCF(nn.Module):
    """
    FIXED: Now inherits from nn.Module for proper parameter handling
    """
    def __init__(self, adj_mat, config=None):
        super().__init__()  # FIXED: Now properly inherits from nn.Module
        
        self.adj_mat = adj_mat
        self.config = config if config else {}
        self.device = self.config.get('device', 'cpu')
        self.n_eigen = self.config.get('n_eigen', 50)
        self.filter_order = self.config.get('filter_order', 3)
        self.lr = self.config.get('lr', 0.01)
        self.filter = self.config.get('filter', 'ui')
        
        # Convert to tensor
        if sp.issparse(self.adj_mat):
            adj_dense = self.adj_mat.toarray()
        else:
            adj_dense = self.adj_mat
            
        self.register_buffer('adj_tensor', torch.tensor(adj_dense, dtype=torch.float32))
        self.n_users, self.n_items = self.adj_tensor.shape
        
        # FIXED: Use the SAME normalization as old model
        row_sums = self.adj_tensor.sum(dim=1, keepdim=True) + 1e-8
        col_sums = self.adj_tensor.sum(dim=0, keepdim=True) + 1e-8
        self.register_buffer('norm_adj', self.adj_tensor / torch.sqrt(row_sums) / torch.sqrt(col_sums))
        
        # Trainable filter modules (will be set in _initialize_model)
        self.user_filter = None
        self.item_filter = None
        
        # Initialize the model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize eigendecompositions and filters"""
        start = time.time()
        
        print(f"Computing eigendecompositions for filter type: {self.filter}")
        
        # Compute similarity matrices
        user_sim = self.norm_adj @ self.norm_adj.t()
        item_sim = self.norm_adj.t() @ self.norm_adj
        
        # Only compute eigendecompositions for selected filters
        if self.filter in ['u', 'ui']:
            user_sim_np = user_sim.cpu().numpy()
            k_user = min(self.n_eigen, self.n_users - 2)
            try:
                eigenvals, eigenvecs = eigsh(sp.csr_matrix(user_sim_np), k=k_user, which='LM')
                self.register_buffer('user_eigenvals', torch.tensor(np.real(eigenvals), dtype=torch.float32))
                self.register_buffer('user_eigenvecs', torch.tensor(np.real(eigenvecs), dtype=torch.float32))
                self.user_filter = UniversalSpectralFilter(self.filter_order)
                print(f"User eigendecomposition: {k_user} components")
            except Exception as e:
                print(f"User eigendecomposition failed: {e}")
                self.register_buffer('user_eigenvals', torch.ones(min(self.n_eigen, self.n_users)))
                self.register_buffer('user_eigenvecs', torch.eye(self.n_users, min(self.n_eigen, self.n_users)))
                self.user_filter = UniversalSpectralFilter(self.filter_order)
        
        if self.filter in ['i', 'ui']:
            item_sim_np = item_sim.cpu().numpy()
            k_item = min(self.n_eigen, self.n_items - 2)
            try:
                eigenvals, eigenvecs = eigsh(sp.csr_matrix(item_sim_np), k=k_item, which='LM')
                self.register_buffer('item_eigenvals', torch.tensor(np.real(eigenvals), dtype=torch.float32))
                self.register_buffer('item_eigenvecs', torch.tensor(np.real(eigenvecs), dtype=torch.float32))
                self.item_filter = UniversalSpectralFilter(self.filter_order)
                print(f"Item eigendecomposition: {k_item} components")
            except Exception as e:
                print(f"Item eigendecomposition failed: {e}")
                self.register_buffer('item_eigenvals', torch.ones(min(self.n_eigen, self.n_items)))
                self.register_buffer('item_eigenvecs', torch.eye(self.n_items, min(self.n_eigen, self.n_items)))
                self.item_filter = UniversalSpectralFilter(self.filter_order)
        
        # FIXED: Set combination weights as nn.Parameter (trainable)
        if self.filter == 'u':
            self.combination_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
        elif self.filter == 'i':
            self.combination_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
        else:  # 'ui'
            self.combination_weights = nn.Parameter(torch.tensor([0.5, 0.3, 0.2]))
        
        end = time.time()
        print(f'Initialization time for Universal-SpectralCF ({self.filter}): {end-start:.2f}s')
    
    def _get_filter_matrices(self):
        """Compute filter matrices without caching for proper gradients"""
        user_matrix = None
        item_matrix = None
        
        if self.filter in ['u', 'ui'] and self.user_filter is not None:
            user_filter_response = self.user_filter(self.user_eigenvals)
            user_matrix = self.user_eigenvecs @ torch.diag(user_filter_response) @ self.user_eigenvecs.t()
        
        if self.filter in ['i', 'ui'] and self.item_filter is not None:
            item_filter_response = self.item_filter(self.item_eigenvals)
            item_matrix = self.item_eigenvecs @ torch.diag(item_filter_response) @ self.item_eigenvecs.t()
        
        return user_matrix, item_matrix
    
    def forward(self, users, target_ratings=None):
        """Forward pass for training"""
        user_profiles = self.adj_tensor[users]
        direct_scores = user_profiles
        
        # Get filter matrices
        user_filter_matrix, item_filter_matrix = self._get_filter_matrices()
        
        # Compute scores based on filter type
        if self.filter == 'u':
            user_filter_rows = user_filter_matrix[users]
            user_scores = user_filter_rows @ self.adj_tensor
            weights = torch.softmax(self.combination_weights, dim=0)
            predicted = weights[0] * direct_scores + weights[1] * user_scores
            
        elif self.filter == 'i':
            item_scores = user_profiles @ item_filter_matrix
            weights = torch.softmax(self.combination_weights, dim=0)
            predicted = weights[0] * direct_scores + weights[1] * item_scores
            
        else:  # 'ui'
            item_scores = user_profiles @ item_filter_matrix
            user_filter_rows = user_filter_matrix[users]
            user_scores = user_filter_rows @ self.adj_tensor
            weights = torch.softmax(self.combination_weights, dim=0)
            predicted = weights[0] * direct_scores + weights[1] * item_scores + weights[2] * user_scores
        
        if target_ratings is not None:
            # Training mode - compute loss
            loss = torch.mean((predicted - target_ratings) ** 2)
            
            # Add regularization
            reg = 0.0
            if self.filter in ['u', 'ui'] and self.user_filter is not None:
                reg += self.user_filter.coeffs.norm(2).pow(2)
            if self.filter in ['i', 'ui'] and self.item_filter is not None:
                reg += self.item_filter.coeffs.norm(2).pow(2)
            reg *= 1e-6
            
            return loss + reg
        else:
            # Evaluation mode - return predictions
            return predicted
        
    def getUsersRating(self, batch_users):
        """Evaluation interface"""
        self.eval()
        with torch.no_grad():
            if isinstance(batch_users, np.ndarray):
                batch_users = torch.LongTensor(batch_users)
            
            # Use forward pass without target_ratings
            combined = self.forward(batch_users, target_ratings=None)
            
        return combined.cpu().numpy()

    def debug_filter_learning(self):
        """Debug what the filters are learning"""
        print("\n=== FILTER LEARNING DEBUG ===")
        with torch.no_grad():
            if self.filter in ['u', 'ui'] and self.user_filter is not None:
                print(f"User filter coefficients: {self.user_filter.coeffs.cpu().numpy()}")
            if self.filter in ['i', 'ui'] and self.item_filter is not None:
                print(f"Item filter coefficients: {self.item_filter.coeffs.cpu().numpy()}")
            
            weights = torch.softmax(self.combination_weights, dim=0)
            print(f"Combination weights: {weights.cpu().numpy()}")
            
            if self.filter in ['u', 'ui'] and self.user_filter is not None:
                user_response = self.user_filter(self.user_eigenvals)
                print(f"User filter response range: [{user_response.min():.4f}, {user_response.max():.4f}]")
            if self.filter in ['i', 'ui'] and self.item_filter is not None:
                item_response = self.item_filter(self.item_eigenvals)
                print(f"Item filter response range: [{item_response.min():.4f}, {item_response.max():.4f}]")
        print("=== END DEBUG ===\n")