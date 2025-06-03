import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import time
from dataloader import BasicDataset
import world

class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError

class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError

class ClosedFormSpectralCF(nn.Module):
    """
    Closed-form spectral CF with fixed polynomial filters (no learning)
    Tests if the spectral approach works conceptually
    """
    
    def __init__(self, adj_mat, config=None):
        super().__init__()
        
        # Convert adjacency matrix to tensor
        if sp.issparse(adj_mat):
            adj_mat = adj_mat.toarray()
        self.adj_mat = torch.tensor(adj_mat, dtype=torch.float32)
        
        self.config = config if config is not None else {}
        self.device = self.config.get("device", "cpu")
        self.n_eigen = self.config.get("n_eigen", 50)
        
        self.n_users, self.n_items = self.adj_mat.shape
        print(f"ClosedFormSpectralCF: {self.n_users} users, {self.n_items} items")
        
        # Move to device
        self.adj_mat = self.adj_mat.to(self.device)
        
        # Simple normalization
        self.norm_adj = None
        self._compute_simple_normalization()
        
        # Fixed filter matrices (computed once)
        self.user_filter_matrix = None
        self.item_filter_matrix = None
        
        # Precompute everything
        self._compute_closed_form_filters()
    
    def _compute_simple_normalization(self):
        """Simple row-column normalization"""
        row_sums = self.adj_mat.sum(dim=1, keepdim=True) + 1e-8
        col_sums = self.adj_mat.sum(dim=0, keepdim=True) + 1e-8
        self.norm_adj = self.adj_mat / torch.sqrt(row_sums) / torch.sqrt(col_sums)
    
    def _fixed_polynomial_filter(self, eigenvalues, filter_type="lowpass"):
        """
        Fixed polynomial filters (no learning)
        """
        # Normalize eigenvalues to [0, 1]
        max_eigenval = torch.max(eigenvalues) + 1e-8
        eigenvals_norm = eigenvalues / max_eigenval
        
        if filter_type == "lowpass":
            # Simple low-pass filter: emphasize small eigenvalues
            filter_response = torch.exp(-eigenvals_norm * 2.0)
        elif filter_type == "bandpass":
            # Band-pass filter: emphasize middle eigenvalues
            filter_response = eigenvals_norm * torch.exp(-eigenvals_norm * 2.0)
        elif filter_type == "highpass":
            # High-pass filter: emphasize large eigenvalues
            filter_response = eigenvals_norm ** 2
        else:  # "identity"
            # Identity filter
            filter_response = torch.ones_like(eigenvals_norm)
        
        return filter_response
    
    def _compute_closed_form_filters(self):
        """
        Compute eigendecompositions and apply fixed filters
        """
        print("Computing eigendecompositions for closed-form filters...")
        
        # Compute similarities from normalized adjacency
        user_similarity = self.norm_adj @ self.norm_adj.t()  # User-user similarity
        item_similarity = self.norm_adj.t() @ self.norm_adj  # Item-item similarity
        
        # Convert to numpy for eigendecomposition
        user_sim_np = user_similarity.detach().cpu().numpy()
        item_sim_np = item_similarity.detach().cpu().numpy()

        n_eigen = self.n_eigen
        
        # User eigendecomposition
        try:
            k_user = min(n_eigen, self.n_users - 2)  # Use more components
            eigenvals, eigenvecs = eigsh(sp.csr_matrix(user_sim_np), k=k_user, which='LM')
            user_eigenvals = torch.tensor(np.real(eigenvals), dtype=torch.float32, device=self.device)
            user_eigenvecs = torch.tensor(np.real(eigenvecs), dtype=torch.float32, device=self.device)
            print(f"User eigendecomposition: {k_user} eigenvalues, range: [{user_eigenvals.min():.4f}, {user_eigenvals.max():.4f}]")
        except Exception as e:
            print(f"User eigendecomposition failed: {e}, using identity")
            user_eigenvals = torch.ones(self.n_users, device=self.device)
            user_eigenvecs = torch.eye(self.n_users, device=self.device)
        
        # Item eigendecomposition
        try:
            k_item = min(n_eigen, self.n_items - 2)
            eigenvals, eigenvecs = eigsh(sp.csr_matrix(item_sim_np), k=k_item, which='LM')
            item_eigenvals = torch.tensor(np.real(eigenvals), dtype=torch.float32, device=self.device)
            item_eigenvecs = torch.tensor(np.real(eigenvecs), dtype=torch.float32, device=self.device)
            print(f"Item eigendecomposition: {k_item} eigenvalues, range: [{item_eigenvals.min():.4f}, {item_eigenvals.max():.4f}]")
        except Exception as e:
            print(f"Item eigendecomposition failed: {e}, using identity")
            item_eigenvals = torch.ones(self.n_items, device=self.device)
            item_eigenvecs = torch.eye(self.n_items, device=self.device)
        
        # Apply fixed filters
        user_filter_response = self._fixed_polynomial_filter(user_eigenvals, "lowpass")
        item_filter_response = self._fixed_polynomial_filter(item_eigenvals, "lowpass")
        
        print(f"User filter response range: [{user_filter_response.min():.4f}, {user_filter_response.max():.4f}]")
        print(f"Item filter response range: [{item_filter_response.min():.4f}, {item_filter_response.max():.4f}]")
        
        # Reconstruct filtered similarity matrices
        if user_eigenvecs.shape[0] == user_eigenvecs.shape[1]:
            # Full eigendecomposition
            self.user_filter_matrix = user_eigenvecs @ torch.diag(user_filter_response) @ user_eigenvecs.t()
        else:
            # Partial eigendecomposition - correct reconstruction
            self.user_filter_matrix = user_eigenvecs @ torch.diag(user_filter_response) @ user_eigenvecs.t()
        
        if item_eigenvecs.shape[0] == item_eigenvecs.shape[1]:
            # Full eigendecomposition
            self.item_filter_matrix = item_eigenvecs @ torch.diag(item_filter_response) @ item_eigenvecs.t()
        else:
            # Partial eigendecomposition - correct reconstruction
            self.item_filter_matrix = item_eigenvecs @ torch.diag(item_filter_response) @ item_eigenvecs.t()
        
        print(f"User filter matrix norm: {self.user_filter_matrix.norm():.4f}")
        print(f"Item filter matrix norm: {self.item_filter_matrix.norm():.4f}")
        print("Closed-form spectral filters computed!")
    
    def forward(self, users, pos_items, neg_items=None):
        """
        Forward pass using fixed spectral filters
        """
        # Get user profiles
        user_profiles = self.adj_mat[users]  # (batch_size, n_items)
        
        # Component 1: Item-based spectral filtering
        item_scores = user_profiles @ self.item_filter_matrix  # (batch_size, n_items)
        
        # Component 2: User-based spectral filtering
        user_filter_rows = self.user_filter_matrix[users]  # (batch_size, n_users)
        user_scores = user_filter_rows @ self.adj_mat  # (batch_size, n_items)
        
        # Component 3: Direct collaborative filtering (baseline)
        direct_scores = user_profiles
        
        # Combine components with fixed weights
        combined_scores = 0.5 * direct_scores + 0.3 * item_scores + 0.2 * user_scores
        
        # Extract scores for positive items
        pos_scores = combined_scores[torch.arange(len(users)), pos_items]
        
        if neg_items is not None:
            # Extract scores for negative items
            neg_scores = combined_scores[torch.arange(len(users)), neg_items]
            return pos_scores, neg_scores
        else:
            return pos_scores
    
    def getUsersRating(self, batch_users):
        """
        Get ratings for all items (evaluation interface)
        """
        self.eval()
        with torch.no_grad():
            # Get user profiles
            user_profiles = self.adj_mat[batch_users]  # (batch_size, n_items)
            
            # Component 1: Item-based spectral filtering
            item_scores = user_profiles @ self.item_filter_matrix  # (batch_size, n_items)
            
            # Component 2: User-based spectral filtering
            user_filter_rows = self.user_filter_matrix[batch_users]  # (batch_size, n_users)
            user_scores = user_filter_rows @ self.adj_mat  # (batch_size, n_items)
            
            # Component 3: Direct collaborative filtering
            direct_scores = user_profiles
            
            # Combine components
            combined_scores = 0.5 * direct_scores + 0.3 * item_scores + 0.2 * user_scores
        
        return combined_scores.cpu().numpy()
    
    def debug_spectral_components(self, test_users=None):
        """
        Debug the different spectral components
        """
        if test_users is None:
            test_users = torch.tensor([0, 1, 2], device=self.device)
        
        print("\n=== SPECTRAL COMPONENTS DEBUG ===")
        
        with torch.no_grad():
            user_profiles = self.adj_mat[test_users]
            
            # Component scores
            direct_scores = user_profiles
            item_scores = user_profiles @ self.item_filter_matrix
            user_filter_rows = self.user_filter_matrix[test_users]
            user_scores = user_filter_rows @ self.adj_mat
            
            print(f"Direct scores stats: mean={direct_scores.mean():.6f}, std={direct_scores.std():.6f}")
            print(f"Item spectral stats: mean={item_scores.mean():.6f}, std={item_scores.std():.6f}")
            print(f"User spectral stats: mean={user_scores.mean():.6f}, std={user_scores.std():.6f}")
            
            # Final combination
            combined = 0.5 * direct_scores + 0.3 * item_scores + 0.2 * user_scores
            print(f"Combined scores stats: mean={combined.mean():.6f}, std={combined.std():.6f}")
        
        print("=== END DEBUG ===\n")

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
        
    def getUsersRating(self, batch_users, ds_name=None):
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