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
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)


class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma

class LGCN_IDE(object):
    def __init__(self, adj_mat):
        self.adj_mat = adj_mat
        
    def train(self):
        adj_mat = self.adj_mat
        start = time.time()
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        d_mat_i = d_mat
        norm_adj = d_mat.dot(adj_mat)

        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        d_mat_u = d_mat
        d_mat_u_inv = sp.diags(1/d_inv)
        norm_adj = norm_adj.dot(d_mat)
        self.norm_adj = norm_adj.tocsr()
        end = time.time()
        print('training time for LGCN-IDE', end-start)
        
    def getUsersRating(self, batch_users, ds_name):
        norm_adj = self.norm_adj
        batch_test = np.array(norm_adj[batch_users,:].todense())
        U_1 = batch_test @ norm_adj.T @ norm_adj
        if(ds_name == 'gowalla'):
            U_2 = U_1 @ norm_adj.T @ norm_adj
            return U_2
        else:
            return U_1
        
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


######################################################################################

# Optimized version with progress bars and performance improvements
# Add these imports to your model.py
from tqdm import tqdm
import time

class UniversalSpectralFilter(nn.Module):
    """
    Learnable universal spectral filter using Chebyshev polynomials
    """
    def __init__(self, filter_order=5):
        super().__init__()
        self.filter_order = filter_order
        
        # Learnable polynomial coefficients - this is what gets trained!
        self.coeffs = nn.Parameter(torch.randn(filter_order + 1) * 0.1)
        
        # Initialize to be roughly low-pass
        with torch.no_grad():
            init_coeffs = torch.tensor([1.0, -0.5, 0.1, -0.02, 0.005, 0.0])
            self.coeffs.data[:len(init_coeffs)] = init_coeffs[:filter_order + 1]
    
    def forward(self, eigenvalues):
        """
        Apply learnable spectral filter using Chebyshev polynomials
        """
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
        
        # Apply activation to ensure positive filter response
        filter_response = torch.relu(filter_response) + 1e-6
        
        return filter_response

class UniversalSpectralCF(PairWiseModel):
    """
    Optimized Universal Spectral Collaborative Filtering with progress tracking
    """
    def __init__(self, config, dataset):
        super().__init__()
        
        self.config = config
        self.dataset = dataset
        
        # Import world here if not available globally
        try:
            import world
            self.device = world.device
        except:
            self.device = config.get('device', 'cpu')
        
        # Get adjacency matrix from dataset
        print("Loading adjacency matrix...")
        start_time = time.time()
        
        adj_mat = dataset.UserItemNet.tolil()
        if sp.issparse(adj_mat):
            print("Converting sparse matrix to dense...")
            adj_mat_dense = adj_mat.toarray()
        else:
            adj_mat_dense = adj_mat
        
        self.adj_mat = torch.tensor(adj_mat_dense, dtype=torch.float32, device=self.device)
        self.n_users, self.n_items = self.adj_mat.shape
        
        print(f"Matrix loaded in {time.time() - start_time:.2f}s")
        
        # Configuration with reduced eigenvalues for speed
        self.n_eigen = min(config.get("n_eigen", 30), 30)  # Limit to 30 for speed
        self.filter_order = config.get("filter_order", 3)  # Reduce order for speed
        
        print(f"UniversalSpectralCF: {self.n_users} users, {self.n_items} items")
        print(f"Using {self.n_eigen} eigenvalues, filter order {self.filter_order}")
        
        # Initialize learnable filters
        self.user_filter = UniversalSpectralFilter(self.filter_order)
        self.item_filter = UniversalSpectralFilter(self.filter_order)
        
        # Learnable combination weights
        self.combination_weights = nn.Parameter(torch.tensor([0.5, 0.3, 0.2]))
        
        # Precomputed matrices (cached for efficiency)
        self.user_eigenvals = None
        self.user_eigenvecs = None
        self.item_eigenvals = None
        self.item_eigenvecs = None
        
        # Cache for reconstructed filter matrices (updated only when needed)
        self._cached_user_filter_matrix = None
        self._cached_item_filter_matrix = None
        self._cache_valid = False
        
        self._precompute_eigendecompositions()
    
    def _precompute_eigendecompositions(self):
        """
        Precompute eigendecompositions with progress tracking
        """
        print("Computing eigendecompositions...")
        total_start = time.time()
        
        # Compute normalization
        print("Computing matrix normalization...")
        norm_start = time.time()
        row_sums = self.adj_mat.sum(dim=1, keepdim=True) + 1e-8
        col_sums = self.adj_mat.sum(dim=0, keepdim=True) + 1e-8
        norm_adj = self.adj_mat / torch.sqrt(row_sums) / torch.sqrt(col_sums)
        print(f"Normalization completed in {time.time() - norm_start:.2f}s")
        
        # Compute similarity matrices
        print("Computing similarity matrices...")
        sim_start = time.time()
        user_similarity = norm_adj @ norm_adj.t()
        item_similarity = norm_adj.t() @ norm_adj
        print(f"Similarity matrices computed in {time.time() - sim_start:.2f}s")
        
        # Convert to numpy for eigendecomposition
        user_sim_np = user_similarity.detach().cpu().numpy()
        item_sim_np = item_similarity.detach().cpu().numpy()
        
        # User eigendecomposition with progress
        print("Computing user eigendecomposition...")
        user_eigen_start = time.time()
        try:
            k_user = min(self.n_eigen, self.n_users - 2)
            eigenvals, eigenvecs = eigsh(sp.csr_matrix(user_sim_np), k=k_user, which='LM')
            self.user_eigenvals = torch.tensor(np.real(eigenvals), dtype=torch.float32, device=self.device)
            self.user_eigenvecs = torch.tensor(np.real(eigenvecs), dtype=torch.float32, device=self.device)
            print(f"User eigendecomposition completed in {time.time() - user_eigen_start:.2f}s ({k_user} components)")
        except Exception as e:
            print(f"User eigendecomposition failed: {e}, using fallback")
            k_user = min(self.n_eigen, self.n_users)
            self.user_eigenvals = torch.ones(k_user, device=self.device)
            self.user_eigenvecs = torch.eye(self.n_users, k_user, device=self.device)
        
        # Item eigendecomposition with progress
        print("Computing item eigendecomposition...")
        item_eigen_start = time.time()
        try:
            k_item = min(self.n_eigen, self.n_items - 2)
            eigenvals, eigenvecs = eigsh(sp.csr_matrix(item_sim_np), k=k_item, which='LM')
            self.item_eigenvals = torch.tensor(np.real(eigenvals), dtype=torch.float32, device=self.device)
            self.item_eigenvecs = torch.tensor(np.real(eigenvecs), dtype=torch.float32, device=self.device)
            print(f"Item eigendecomposition completed in {time.time() - item_eigen_start:.2f}s ({k_item} components)")
        except Exception as e:
            print(f"Item eigendecomposition failed: {e}, using fallback")
            k_item = min(self.n_eigen, self.n_items)
            self.item_eigenvals = torch.ones(k_item, device=self.device)
            self.item_eigenvecs = torch.eye(self.n_items, k_item, device=self.device)
        
        print(f"Total eigendecomposition time: {time.time() - total_start:.2f}s")
        print("Eigendecompositions completed!")
    
    def _get_filter_matrices(self):
        """
        Get filter matrices with caching for efficiency
        """
        if not self._cache_valid:
            # Recompute filter matrices only when coefficients change
            user_filter_response = self.user_filter(self.user_eigenvals)
            item_filter_response = self.item_filter(self.item_eigenvals)
            
            self._cached_user_filter_matrix = self.user_eigenvecs @ torch.diag(user_filter_response) @ self.user_eigenvecs.t()
            self._cached_item_filter_matrix = self.item_eigenvecs @ torch.diag(item_filter_response) @ self.item_eigenvecs.t()
            
            self._cache_valid = True
        
        return self._cached_user_filter_matrix, self._cached_item_filter_matrix
    
    def _invalidate_cache(self):
        """Invalidate cache when parameters change"""
        self._cache_valid = False
    
    def forward(self, users, items):
        """
        Forward pass for prediction (used in evaluation)
        """
        user_filter_matrix, item_filter_matrix = self._get_filter_matrices()
        
        # Get user profiles
        user_profiles = self.adj_mat[users]
        
        # Apply spectral filtering
        direct_scores = user_profiles
        item_scores = user_profiles @ item_filter_matrix
        user_filter_rows = user_filter_matrix[users]
        user_scores = user_filter_rows @ self.adj_mat
        
        # Combine with learnable weights
        weights = torch.softmax(self.combination_weights, dim=0)
        combined_scores = (weights[0] * direct_scores + 
                          weights[1] * item_scores + 
                          weights[2] * user_scores)
        
        # Extract scores for specific items
        item_scores_final = combined_scores[torch.arange(len(users)), items]
        
        return torch.sigmoid(item_scores_final)
    
    def bpr_loss(self, users, pos, neg):
        """
        BPR loss implementation with cache invalidation
        """
        # Invalidate cache since we're in training mode
        self._invalidate_cache()
        
        # Apply learnable spectral filters
        user_filter_response = self.user_filter(self.user_eigenvals)
        item_filter_response = self.item_filter(self.item_eigenvals)
        
        # Reconstruct filtered matrices
        user_filter_matrix = self.user_eigenvecs @ torch.diag(user_filter_response) @ self.user_eigenvecs.t()
        item_filter_matrix = self.item_eigenvecs @ torch.diag(item_filter_response) @ self.item_eigenvecs.t()
        
        # Get user profiles for this batch
        batch_user_profiles = self.adj_mat[users]
        
        # Apply spectral filtering
        direct_scores = batch_user_profiles
        item_scores = batch_user_profiles @ item_filter_matrix
        user_filter_rows = user_filter_matrix[users]
        user_scores = user_filter_rows @ self.adj_mat
        
        # Combine with learnable weights
        weights = torch.softmax(self.combination_weights, dim=0)
        combined_scores = (weights[0] * direct_scores + 
                          weights[1] * item_scores + 
                          weights[2] * user_scores)
        
        # Extract scores for positive and negative items
        pos_scores = combined_scores[torch.arange(len(users)), pos]
        neg_scores = combined_scores[torch.arange(len(users)), neg]
        
        # BPR loss
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        # L2 regularization on filter coefficients
        reg_loss = (self.user_filter.coeffs.norm(2).pow(2) + 
                   self.item_filter.coeffs.norm(2).pow(2) + 
                   self.combination_weights.norm(2).pow(2)) * 1e-5
        
        return loss, reg_loss
    
    def getUsersRating(self, users):
        """
        Get ratings for all items (evaluation interface) with caching
        """
        self.eval()
        with torch.no_grad():
            user_filter_matrix, item_filter_matrix = self._get_filter_matrices()
            
            # Compute scores for all items
            batch_user_profiles = self.adj_mat[users]
            
            direct_scores = batch_user_profiles
            item_scores = batch_user_profiles @ item_filter_matrix
            user_filter_rows = user_filter_matrix[users]
            user_scores = user_filter_rows @ self.adj_mat
            
            weights = torch.softmax(self.combination_weights, dim=0)
            combined_scores = (weights[0] * direct_scores + 
                             weights[1] * item_scores + 
                             weights[2] * user_scores)
        
        return combined_scores
    
    def debug_filter_learning(self):
        """
        Debug what the filters are learning
        """
        print("\n=== FILTER LEARNING DEBUG ===")
        with torch.no_grad():
            print(f"User filter coefficients: {self.user_filter.coeffs.cpu().numpy()}")
            print(f"Item filter coefficients: {self.item_filter.coeffs.cpu().numpy()}")
            
            weights = torch.softmax(self.combination_weights, dim=0)
            print(f"Combination weights: {weights.cpu().numpy()}")
            
            # Sample filter responses
            user_response = self.user_filter(self.user_eigenvals)
            item_response = self.item_filter(self.item_eigenvals)
            
            print(f"User filter response range: [{user_response.min():.4f}, {user_response.max():.4f}]")
            print(f"Item filter response range: [{item_response.min():.4f}, {item_response.max():.4f}]")
        print("=== END DEBUG ===\n")
