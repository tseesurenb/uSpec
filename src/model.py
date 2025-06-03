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


        
class GF_CF(object):
    def __init__(self, adj_mat):
        self.adj_mat = adj_mat
        
    def train(self):
        adj_mat = self.adj_mat
        start = time.time()
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)

        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        self.d_mat_i = d_mat
        self.d_mat_i_inv = sp.diags(1/d_inv)
        norm_adj = norm_adj.dot(d_mat)
        self.norm_adj = norm_adj.tocsc()
        ut, s, self.vt = sparsesvd(self.norm_adj, 256)
        end = time.time()
        print('training time for GF-CF', end-start)
        
    def getUsersRating(self, batch_users, ds_name):
        norm_adj = self.norm_adj
        adj_mat = self.adj_mat
        batch_test = np.array(adj_mat[batch_users,:].todense())
        U_2 = batch_test @ norm_adj.T @ norm_adj
        if(ds_name == 'amazon-book'):
            ret = U_2
        else:
            U_1 = batch_test @  self.d_mat_i @ self.vt.T @ self.vt @ self.d_mat_i_inv
            ret = U_2 + 0.3 * U_1
        return ret
        
class ClosedFormSpectralCF(object):
    def __init__(self, adj_mat):
        self.adj_mat = adj_mat
        self.device = 'cpu'
        self.n_eigen = 50
        
    def train(self):
        start = time.time()
        if sp.issparse(self.adj_mat):
            adj_dense = self.adj_mat.toarray()
        else:
            adj_dense = self.adj_mat
        
        self.adj_tensor = torch.tensor(adj_dense, dtype=torch.float32)
        self.n_users, self.n_items = self.adj_tensor.shape
        
        row_sums = self.adj_tensor.sum(dim=1, keepdim=True) + 1e-8
        col_sums = self.adj_tensor.sum(dim=0, keepdim=True) + 1e-8
        norm_adj = self.adj_tensor / torch.sqrt(row_sums) / torch.sqrt(col_sums)
        
        user_sim = norm_adj @ norm_adj.t()
        item_sim = norm_adj.t() @ norm_adj
        
        user_sim_np = user_sim.numpy()
        item_sim_np = item_sim.numpy()
        
        k_user = min(self.n_eigen, self.n_users - 2)
        eigenvals, eigenvecs = eigsh(sp.csr_matrix(user_sim_np), k=k_user, which='LM')
        user_eigenvals = torch.tensor(np.real(eigenvals), dtype=torch.float32)
        user_eigenvecs = torch.tensor(np.real(eigenvecs), dtype=torch.float32)
        
        k_item = min(self.n_eigen, self.n_items - 2)
        eigenvals, eigenvecs = eigsh(sp.csr_matrix(item_sim_np), k=k_item, which='LM')
        item_eigenvals = torch.tensor(np.real(eigenvals), dtype=torch.float32)
        item_eigenvecs = torch.tensor(np.real(eigenvecs), dtype=torch.float32)
        
        max_eval = torch.max(user_eigenvals) + 1e-8
        user_filter = torch.exp(-user_eigenvals / max_eval * 2.0)
        max_eval = torch.max(item_eigenvals) + 1e-8
        item_filter = torch.exp(-item_eigenvals / max_eval * 2.0)
        
        self.user_filter_matrix = user_eigenvecs @ torch.diag(user_filter) @ user_eigenvecs.t()
        self.item_filter_matrix = item_eigenvecs @ torch.diag(item_filter) @ item_eigenvecs.t()
        
        end = time.time()
        print('training time for ClosedForm-SpectralCF', end-start)
        
    def getUsersRating(self, batch_users, ds_name):
        user_profiles = self.adj_tensor[batch_users]
        item_scores = user_profiles @ self.item_filter_matrix
        user_filter_rows = self.user_filter_matrix[batch_users]
        user_scores = user_filter_rows @ self.adj_tensor
        direct_scores = user_profiles
        combined = 0.5 * direct_scores + 0.3 * item_scores + 0.2 * user_scores
        return combined.numpy()


class UniversalSpectralFilter(nn.Module):
    def __init__(self, filter_order=3):
        super().__init__()
        self.coeffs = nn.Parameter(torch.tensor([1.0, -0.5, 0.1, 0.0][:filter_order + 1]))
    
    def forward(self, eigenvalues):
        max_eigenval = torch.max(eigenvalues) + 1e-8
        x = 2 * (eigenvalues / max_eigenval) - 1
        result = self.coeffs[0] * torch.ones_like(x)
        if len(self.coeffs) > 1:
            T_prev = torch.ones_like(x)
            T_curr = x
            result += self.coeffs[1] * T_curr
            for i in range(2, len(self.coeffs)):
                T_next = 2 * x * T_curr - T_prev
                result += self.coeffs[i] * T_next
                T_prev, T_curr = T_curr, T_next
        return torch.relu(result) + 1e-6


class UniversalSpectralCF(object):
    def __init__(self, adj_mat, config=None):
        self.adj_mat = adj_mat
        self.config = config if config else {}
        self.device = self.config.get('device', 'cpu')
        self.n_eigen = self.config.get('n_eigen', 20)
        self.filter_order = self.config.get('filter_order', 2)
        self.lr = self.config.get('lr', 0.01)
        self.filter = self.config.get('filter', 'ui')  # 'u', 'i', or 'ui'
        
    def train(self):
        start = time.time()
        if sp.issparse(self.adj_mat):
            adj_dense = self.adj_mat.toarray()
        else:
            adj_dense = self.adj_mat
            
        self.adj_tensor = torch.tensor(adj_dense, dtype=torch.float32)
        self.n_users, self.n_items = self.adj_tensor.shape
        
        row_sums = self.adj_tensor.sum(dim=1, keepdim=True) + 1e-8
        col_sums = self.adj_tensor.sum(dim=0, keepdim=True) + 1e-8
        norm_adj = self.adj_tensor / torch.sqrt(row_sums) / torch.sqrt(col_sums)
        
        # Only compute eigendecompositions for selected filters
        if self.filter in ['u', 'ui']:
            user_sim = norm_adj @ norm_adj.t()
            user_sim_np = user_sim.numpy()
            k_user = min(self.n_eigen, self.n_users - 2)
            eigenvals, eigenvecs = eigsh(sp.csr_matrix(user_sim_np), k=k_user, which='LM')
            self.user_eigenvals = torch.tensor(np.real(eigenvals), dtype=torch.float32)
            self.user_eigenvecs = torch.tensor(np.real(eigenvecs), dtype=torch.float32)
            self.user_filter = UniversalSpectralFilter(self.filter_order)
        
        if self.filter in ['i', 'ui']:
            item_sim = norm_adj.t() @ norm_adj
            item_sim_np = item_sim.numpy()
            k_item = min(self.n_eigen, self.n_items - 2)
            eigenvals, eigenvecs = eigsh(sp.csr_matrix(item_sim_np), k=k_item, which='LM')
            self.item_eigenvals = torch.tensor(np.real(eigenvals), dtype=torch.float32)
            self.item_eigenvecs = torch.tensor(np.real(eigenvecs), dtype=torch.float32)
            self.item_filter = UniversalSpectralFilter(self.filter_order)
        
        # Set combination weights based on filter type
        if self.filter == 'u':
            self.combination_weights = nn.Parameter(torch.tensor([0.5, 0.5]))  # [direct, user]
        elif self.filter == 'i':
            self.combination_weights = nn.Parameter(torch.tensor([0.5, 0.5]))  # [direct, item]
        else:  # 'ui'
            self.combination_weights = nn.Parameter(torch.tensor([0.4, 0.3, 0.3]))  # [direct, item, user]
        
        # Setup optimizer with only relevant parameters
        params = [self.combination_weights]
        if self.filter in ['u', 'ui']:
            params.extend(list(self.user_filter.parameters()))
        if self.filter in ['i', 'ui']:
            params.extend(list(self.item_filter.parameters()))
        self.optimizer = torch.optim.Adam(params, lr=self.lr)
        
        end = time.time()
        print(f'training time for Universal-SpectralCF ({self.filter})', end-start)
        
    def train_step(self, users, target_ratings):
        self.optimizer.zero_grad()
        
        # Convert inputs to tensors if they're numpy arrays
        if isinstance(users, np.ndarray):
            users = torch.LongTensor(users)
        if isinstance(target_ratings, np.ndarray):
            target_ratings = torch.FloatTensor(target_ratings)
        
        user_profiles = self.adj_tensor[users]
        direct_scores = user_profiles
        
        # Compute scores based on filter type
        if self.filter == 'u':
            user_response = self.user_filter(self.user_eigenvals)
            user_filter_matrix = self.user_eigenvecs @ torch.diag(user_response) @ self.user_eigenvecs.t()
            user_filter_rows = user_filter_matrix[users]
            user_scores = user_filter_rows @ self.adj_tensor
            weights = torch.softmax(self.combination_weights, dim=0)
            predicted = weights[0] * direct_scores + weights[1] * user_scores
            reg = self.user_filter.coeffs.norm(2).pow(2) * 1e-6
            
        elif self.filter == 'i':
            item_response = self.item_filter(self.item_eigenvals)
            item_filter_matrix = self.item_eigenvecs @ torch.diag(item_response) @ self.item_eigenvecs.t()
            item_scores = user_profiles @ item_filter_matrix
            weights = torch.softmax(self.combination_weights, dim=0)
            predicted = weights[0] * direct_scores + weights[1] * item_scores
            reg = self.item_filter.coeffs.norm(2).pow(2) * 1e-6
            
        else:  # 'ui'
            user_response = self.user_filter(self.user_eigenvals)
            item_response = self.item_filter(self.item_eigenvals)
            user_filter_matrix = self.user_eigenvecs @ torch.diag(user_response) @ self.user_eigenvecs.t()
            item_filter_matrix = self.item_eigenvecs @ torch.diag(item_response) @ self.item_eigenvecs.t()
            item_scores = user_profiles @ item_filter_matrix
            user_filter_rows = user_filter_matrix[users]
            user_scores = user_filter_rows @ self.adj_tensor
            weights = torch.softmax(self.combination_weights, dim=0)
            predicted = weights[0] * direct_scores + weights[1] * item_scores + weights[2] * user_scores
            reg = (self.user_filter.coeffs.norm(2).pow(2) + self.item_filter.coeffs.norm(2).pow(2)) * 1e-6
        
        loss = torch.mean((predicted - target_ratings) ** 2)
        total_loss = loss + reg
        
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach().item()  # Use detach() before item()
        
    def getUsersRating(self, batch_users, ds_name = None):
        with torch.no_grad():
            user_profiles = self.adj_tensor[batch_users]
            direct_scores = user_profiles
            
            # Compute scores based on filter type
            if self.filter == 'u':
                user_response = self.user_filter(self.user_eigenvals)
                user_filter_matrix = self.user_eigenvecs @ torch.diag(user_response) @ self.user_eigenvecs.t()
                user_filter_rows = user_filter_matrix[batch_users]
                user_scores = user_filter_rows @ self.adj_tensor
                weights = torch.softmax(self.combination_weights, dim=0)
                combined = weights[0] * direct_scores + weights[1] * user_scores
                
            elif self.filter == 'i':
                item_response = self.item_filter(self.item_eigenvals)
                item_filter_matrix = self.item_eigenvecs @ torch.diag(item_response) @ self.item_eigenvecs.t()
                item_scores = user_profiles @ item_filter_matrix
                weights = torch.softmax(self.combination_weights, dim=0)
                combined = weights[0] * direct_scores + weights[1] * item_scores
                
            else:  # 'ui'
                user_response = self.user_filter(self.user_eigenvals)
                item_response = self.item_filter(self.item_eigenvals)
                user_filter_matrix = self.user_eigenvecs @ torch.diag(user_response) @ self.user_eigenvecs.t()
                item_filter_matrix = self.item_eigenvecs @ torch.diag(item_response) @ self.item_eigenvecs.t()
                item_scores = user_profiles @ item_filter_matrix
                user_filter_rows = user_filter_matrix[batch_users]
                user_scores = user_filter_rows @ self.adj_tensor
                weights = torch.softmax(self.combination_weights, dim=0)
                combined = weights[0] * direct_scores + weights[1] * item_scores + weights[2] * user_scores
            
        return combined.numpy()

    def debug_filter_learning(self):
        """Debug what the filters are learning"""
        print("\n=== FILTER LEARNING DEBUG ===")
        with torch.no_grad():
            if self.filter in ['u', 'ui']:
                print(f"User filter coefficients: {self.user_filter.coeffs.cpu().numpy()}")
            if self.filter in ['i', 'ui']:
                print(f"Item filter coefficients: {self.item_filter.coeffs.cpu().numpy()}")
            
            weights = torch.softmax(self.combination_weights, dim=0)
            print(f"Combination weights: {weights.cpu().numpy()}")
            
            if self.filter in ['u', 'ui']:
                user_response = self.user_filter(self.user_eigenvals)
                print(f"User filter response range: [{user_response.min():.4f}, {user_response.max():.4f}]")
            if self.filter in ['i', 'ui']:
                item_response = self.item_filter(self.item_eigenvals)
                print(f"Item filter response range: [{item_response.min():.4f}, {item_response.max():.4f}]")
        print("=== END DEBUG ===\n")