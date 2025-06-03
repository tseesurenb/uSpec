'''
Created on June 3, 2025
Pytorch Implementation of uSpec with Positive and Negative Similarities
Extended from Universal Spectral Collaborative Filtering

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
        filter_response = torch.exp(-torch.abs(filter_response)) + 1e-6
        
        return filter_response

class UniversalSpectralCF(nn.Module):
    """
    Universal Spectral CF with Positive and Negative Similarities
    """
    def __init__(self, adj_mat, config=None):
        super().__init__()
        
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
        
        # Trainable filter modules (will be set in _initialize_model)
        self.user_pos_filter = None
        self.user_neg_filter = None
        self.item_pos_filter = None
        self.item_neg_filter = None
        
        # Initialize the model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize eigendecompositions and filters for both positive and negative similarities"""
        start = time.time()
        
        print(f"Computing positive and negative eigendecompositions for filter type: {self.filter}")
        
        # Compute positive similarity matrices
        user_pos_sim = self.norm_adj @ self.norm_adj.t()
        item_pos_sim = self.norm_adj.t() @ self.norm_adj
        
        # Compute negative similarity matrices
        user_neg_sim = self.neg_norm_adj @ self.neg_norm_adj.t()
        item_neg_sim = self.neg_norm_adj.t() @ self.neg_norm_adj
        
        # Initialize positive and negative filters for users
        if self.filter in ['u', 'ui']:
            # Positive user similarities
            user_pos_sim_np = user_pos_sim.cpu().numpy()
            k_user = min(self.n_eigen, self.n_users - 2)
            try:
                eigenvals, eigenvecs = eigsh(sp.csr_matrix(user_pos_sim_np), k=k_user, which='LM')
                self.register_buffer('user_pos_eigenvals', torch.tensor(np.real(eigenvals), dtype=torch.float32))
                self.register_buffer('user_pos_eigenvecs', torch.tensor(np.real(eigenvecs), dtype=torch.float32))
                self.user_pos_filter = UniversalSpectralFilter(self.filter_order)
                print(f"User positive eigendecomposition: {k_user} components")
            except Exception as e:
                print(f"User positive eigendecomposition failed: {e}")
                self.register_buffer('user_pos_eigenvals', torch.ones(k_user))
                self.register_buffer('user_pos_eigenvecs', torch.eye(self.n_users, k_user))
                self.user_pos_filter = UniversalSpectralFilter(self.filter_order)
            
            # Negative user similarities
            user_neg_sim_np = user_neg_sim.cpu().numpy()
            try:
                eigenvals, eigenvecs = eigsh(sp.csr_matrix(user_neg_sim_np), k=k_user, which='LM')
                self.register_buffer('user_neg_eigenvals', torch.tensor(np.real(eigenvals), dtype=torch.float32))
                self.register_buffer('user_neg_eigenvecs', torch.tensor(np.real(eigenvecs), dtype=torch.float32))
                self.user_neg_filter = UniversalSpectralFilter(self.filter_order)
                print(f"User negative eigendecomposition: {k_user} components")
            except Exception as e:
                print(f"User negative eigendecomposition failed: {e}")
                self.register_buffer('user_neg_eigenvals', torch.ones(k_user))
                self.register_buffer('user_neg_eigenvecs', torch.eye(self.n_users, k_user))
                self.user_neg_filter = UniversalSpectralFilter(self.filter_order)
        
        # Initialize positive and negative filters for items
        if self.filter in ['i', 'ui']:
            # Positive item similarities
            item_pos_sim_np = item_pos_sim.cpu().numpy()
            k_item = min(self.n_eigen, self.n_items - 2)
            try:
                eigenvals, eigenvecs = eigsh(sp.csr_matrix(item_pos_sim_np), k=k_item, which='LM')
                self.register_buffer('item_pos_eigenvals', torch.tensor(np.real(eigenvals), dtype=torch.float32))
                self.register_buffer('item_pos_eigenvecs', torch.tensor(np.real(eigenvecs), dtype=torch.float32))
                self.item_pos_filter = UniversalSpectralFilter(self.filter_order)
                print(f"Item positive eigendecomposition: {k_item} components")
            except Exception as e:
                print(f"Item positive eigendecomposition failed: {e}")
                self.register_buffer('item_pos_eigenvals', torch.ones(k_item))
                self.register_buffer('item_pos_eigenvecs', torch.eye(self.n_items, k_item))
                self.item_pos_filter = UniversalSpectralFilter(self.filter_order)
            
            # Negative item similarities
            item_neg_sim_np = item_neg_sim.cpu().numpy()
            try:
                eigenvals, eigenvecs = eigsh(sp.csr_matrix(item_neg_sim_np), k=k_item, which='LM')
                self.register_buffer('item_neg_eigenvals', torch.tensor(np.real(eigenvals), dtype=torch.float32))
                self.register_buffer('item_neg_eigenvecs', torch.tensor(np.real(eigenvecs), dtype=torch.float32))
                self.item_neg_filter = UniversalSpectralFilter(self.filter_order)
                print(f"Item negative eigendecomposition: {k_item} components")
            except Exception as e:
                print(f"Item negative eigendecomposition failed: {e}")
                self.register_buffer('item_neg_eigenvals', torch.ones(k_item))
                self.register_buffer('item_neg_eigenvecs', torch.eye(self.n_items, k_item))
                self.item_neg_filter = UniversalSpectralFilter(self.filter_order)
        
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
        print(f'Initialization time for Universal-SpectralCF with pos/neg similarities ({self.filter}): {end-start:.2f}s')
    
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
    
    def forward(self, users, target_ratings=None):
        """Forward pass for training with positive and negative similarities"""
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
        
        if target_ratings is not None:
            # Training mode - compute loss
            loss = torch.mean((predicted - target_ratings) ** 2)
            
            # Add regularization for all filters
            reg = 0.0
            if self.filter in ['u', 'ui']:
                if self.user_pos_filter is not None:
                    reg += self.user_pos_filter.coeffs.norm(2).pow(2)
                if self.user_neg_filter is not None:
                    reg += self.user_neg_filter.coeffs.norm(2).pow(2)
            if self.filter in ['i', 'ui']:
                if self.item_pos_filter is not None:
                    reg += self.item_pos_filter.coeffs.norm(2).pow(2)
                if self.item_neg_filter is not None:
                    reg += self.item_neg_filter.coeffs.norm(2).pow(2)
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
        print("\n=== FILTER LEARNING DEBUG (POS/NEG) ===")
        with torch.no_grad():
            if self.filter in ['u', 'ui']:
                if self.user_pos_filter is not None:
                    print(f"User positive filter coefficients: {self.user_pos_filter.coeffs.cpu().numpy()}")
                if self.user_neg_filter is not None:
                    print(f"User negative filter coefficients: {self.user_neg_filter.coeffs.cpu().numpy()}")
            
            if self.filter in ['i', 'ui']:
                if self.item_pos_filter is not None:
                    print(f"Item positive filter coefficients: {self.item_pos_filter.coeffs.cpu().numpy()}")
                if self.item_neg_filter is not None:
                    print(f"Item negative filter coefficients: {self.item_neg_filter.coeffs.cpu().numpy()}")
            
            weights = torch.softmax(self.combination_weights, dim=0)
            print(f"Combination weights: {weights.cpu().numpy()}")
            
            if self.filter in ['u', 'ui']:
                if self.user_pos_filter is not None:
                    user_pos_response = self.user_pos_filter(self.user_pos_eigenvals)
                    print(f"User positive filter response range: [{user_pos_response.min():.4f}, {user_pos_response.max():.4f}]")
                if self.user_neg_filter is not None:
                    user_neg_response = self.user_neg_filter(self.user_neg_eigenvals)
                    print(f"User negative filter response range: [{user_neg_response.min():.4f}, {user_neg_response.max():.4f}]")
            
            if self.filter in ['i', 'ui']:
                if self.item_pos_filter is not None:
                    item_pos_response = self.item_pos_filter(self.item_pos_eigenvals)
                    print(f"Item positive filter response range: [{item_pos_response.min():.4f}, {item_pos_response.max():.4f}]")
                if self.item_neg_filter is not None:
                    item_neg_response = self.item_neg_filter(self.item_neg_eigenvals)
                    print(f"Item negative filter response range: [{item_neg_response.min():.4f}, {item_neg_response.max():.4f}]")
        print("=== END DEBUG ===\n")