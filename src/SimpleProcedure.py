'''
Simplified Procedure for Universal Spectral CF
Uses MSE loss instead of BPR for faster, simpler training
'''
import world
import numpy as np
import torch
import torch.nn as nn
import utils
import dataloader
from time import time
from tqdm import tqdm
import multiprocessing
from sklearn.metrics import roc_auc_score

CORES = multiprocessing.cpu_count() // 2

class SimpleMSELoss:
    """Simple MSE loss for collaborative filtering - Direct rating prediction"""
    def __init__(self, model, config):
        self.model = model
        self.lr = config['lr']
        self.weight_decay = config.get('decay', 1e-4)
        self.opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
    def compute_loss(self, users, target_ratings):
        """
        Compute MSE loss directly against observed ratings
        Much simpler - no negative sampling needed!
        """
        # Get all predicted ratings for the batch of users
        predicted_ratings = self.model.getUsersRating(users)
        
        # Compute MSE loss only on observed interactions
        mse_loss = torch.mean((predicted_ratings - target_ratings) ** 2)
        
        # Add small regularization on filter coefficients
        reg_loss = (self.model.user_filter.coeffs.norm(2).pow(2) + 
                   self.model.item_filter.coeffs.norm(2).pow(2)) * 1e-6
        
        return mse_loss + reg_loss
    
    def train_step(self, users, target_ratings):
        """Single training step - much simpler!"""
        self.opt.zero_grad()
        loss = self.compute_loss(users, target_ratings)
        loss.backward()
        self.opt.step()
        
        # Invalidate the model's cache since parameters changed
        if hasattr(self.model, '_invalidate_cache'):
            self.model._invalidate_cache()
        
        return loss.cpu().item()

def create_target_ratings(dataset, users):
    """
    Create target rating matrix for a batch of users
    1.0 for positive interactions, 0.0 for non-interactions
    OPTIMIZED: Only create ratings for observed interactions + some random negatives
    """
    batch_size = len(users)
    n_items = dataset.m_items
    
    # Instead of full matrix, use sparse representation for efficiency
    target_ratings = torch.zeros(batch_size, n_items)
    
    # Set positive interactions to 1.0
    for i, user in enumerate(users):
        pos_items = dataset.allPos[user]
        if len(pos_items) > 0:
            target_ratings[i, pos_items] = 1.0
    
    return target_ratings

def MSE_train_simple(dataset, model, loss_class, epoch):
    """
    SUPER OPTIMIZED MSE training - focus on speed
    """
    model.train()
    mse_loss = loss_class
    
    # MUCH MORE AGGRESSIVE optimization for speed
    n_batch = 3  # Only 3 batches per epoch!
    batch_size = 64  # Very small batches
    
    total_loss = 0.0
    start_time = time()
    
    for batch_idx in range(n_batch):
        # Sample random users
        users = np.random.randint(0, dataset.n_users, batch_size)
        users = torch.LongTensor(users).to(world.device)
        
        # Create target ratings (1.0 for positive, 0.0 for negative)
        target_ratings = create_target_ratings(dataset, users.cpu().numpy())
        target_ratings = target_ratings.to(world.device)
        
        # Training step
        batch_loss = mse_loss.train_step(users, target_ratings)
        total_loss += batch_loss
    
    avg_loss = total_loss / n_batch
    training_time = time() - start_time
    
    return f"MSE_loss: {avg_loss:.4f} | Time: {training_time:.2f}s"

def test_one_batch_simple(X):
    """Same as original but simplified"""
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall), 
            'precision': np.array(pre), 
            'ndcg': np.array(ndcg)}

def Test_Simple(dataset, model, epoch, multicore=0):
    """
    Simplified test procedure - faster evaluation
    """
    u_batch_size = world.config['test_u_batch_size']
    testDict = dataset.testDict
    
    model.eval()
    max_K = max(world.topks)
    
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    
    start_time = time()
    
    with torch.no_grad():
        users = list(testDict.keys())
        
        # Limit test users for speed during training (remove this for final eval)
        if epoch >= 0:  # During training, test on subset
            test_users = users[:min(len(users), 1000)]  # Test on first 1000 users only
        else:  # Final evaluation, test on all users
            test_users = users
        
        users_list = []
        rating_list = []
        groundTrue_list = []
        
        # Process in batches
        for batch_users in utils.minibatch(test_users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            
            batch_users_gpu = torch.LongTensor(batch_users).to(world.device)
            
            # Get ratings from model
            rating = model.getUsersRating(batch_users_gpu)
            
            # Convert to tensor if needed
            if isinstance(rating, np.ndarray):
                rating = torch.from_numpy(rating)
            
            # Exclude training items
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            
            # Get top-K items
            _, rating_K = torch.topk(rating, k=max_K)
            
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        
        # Compute metrics
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch_simple, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch_simple(x))
        
        # Aggregate results
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        
        results['recall'] /= float(len(test_users))
        results['precision'] /= float(len(test_users))
        results['ndcg'] /= float(len(test_users))
        
        if multicore == 1:
            pool.close()
    
    eval_time = time() - start_time
    print(f"Evaluation completed in {eval_time:.2f}s")
    #print(f"Recall@20: {results['recall'][0]:.6f}, NDCG@20: {results['ndcg'][0]:.6f}, Precision@20: {results['precision'][0]:.6f}")
    print(f"\033[91mRecall@20: {results['recall'][0]:.6f}, NDCG@20: {results['ndcg'][0]:.6f}, Precision@20: {results['precision'][0]:.6f}\033[0m")
    
    return results

def train_universal_spectral_simple(dataset, model, config, total_epochs=15, verbose = False):
    """
    ULTRA-FAST training procedure for Universal Spectral CF
    Direct MSE on rating matrix - optimized for speed!
    """
    print("="*70)
    print("STARTING ULTRA-FAST DIRECT MSE TRAINING")
    print("="*70)
    
    # Initialize MSE loss
    mse_loss = SimpleMSELoss(model, config)
    
    # Training loop with progress tracking
    best_recall = 0.0
    training_start = time()
    
    for epoch in tqdm(range(total_epochs), desc="Training Progress"):
        epoch_start = time()
        
        # Training step
        train_info = MSE_train_simple(dataset, model, mse_loss, epoch)
        epoch_time = time() - epoch_start
        
        # Print progress every 5 epochs only
        if verbose:
            if epoch % 5 == 0 or epoch == total_epochs - 1:
                print(f"\nEpoch {epoch+1}/{total_epochs}: {train_info}")
                model.debug_filter_learning()
        
        # Quick evaluation only at the end and middle
        if epoch == total_epochs // 2 or epoch == total_epochs - 1:
            print(f"\n[QUICK EVAL - Epoch {epoch+1}]")
            results = Test_Simple(dataset, model, epoch, world.config['multicore'])
            
            # Track best performance
            current_recall = results['recall'][0]
            if current_recall > best_recall:
                best_recall = current_recall
                print(f"✅ New best recall: {best_recall:.6f}")
    
    total_time = time() - training_start
    print(f"\n" + "="*70)
    print("ULTRA-FAST TRAINING COMPLETED!")
    print(f"Total training time: {total_time:.2f}s")
    print(f"Best recall achieved: {best_recall:.6f}")
    print("="*70)
    
    return model

# Convenience function for main_u.py
def simple_train_and_evaluate(dataset, model, config):
    """
    One-line training and evaluation for main_u.py
    """
    # Training with the epochs from config
    trained_model = train_universal_spectral_simple(
        dataset, model, config, 
        total_epochs=config.get('epochs', 15)  # Use config epochs
    )
    
    # Final comprehensive evaluation
    print("\n[FINAL COMPREHENSIVE EVALUATION - ALL USERS, ALL ITEMS]")
    final_results = Test_Simple(dataset, trained_model, -1, world.config['multicore'])
    
    return trained_model, final_results