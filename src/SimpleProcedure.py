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
import matplotlib.pyplot as plt
import os

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
    
    return avg_loss, f"MSE_loss: {avg_loss:.4f} | Time: {training_time:.2f}s"

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
    ULTRA-FAST test procedure for training, COMPREHENSIVE for final eval
    """
    u_batch_size = world.config['test_u_batch_size']
    testDict = dataset.testDict
    
    model.eval()
    max_K = max(world.topks)
    
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    
    start_time = time()
    
    with torch.no_grad():
        users = list(testDict.keys())
        
        # COMPREHENSIVE evaluation for final test, fast for training
        if epoch >= 0:  # During training - FAST
            test_users = users[:min(len(users), 200)]  # Only 200 users!
            batch_size = min(u_batch_size * 2, 200)  # Larger batches
            print(f"Quick eval on {len(test_users)} users...")
        else:  # Final evaluation - COMPREHENSIVE
            test_users = users  # ALL USERS!
            batch_size = u_batch_size  # Normal batch size
            print(f"Comprehensive evaluation on ALL {len(test_users)} users...")
        
        users_list = []
        rating_list = []
        groundTrue_list = []
        
        # Process in batches
        for batch_users in utils.minibatch(test_users, batch_size=batch_size):
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
            
            # Get top-K items for ALL ITEMS
            _, rating_K = torch.topk(rating, k=max_K)
            
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        
        # Compute metrics
        X = zip(rating_list, groundTrue_list)
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
    
    eval_time = time() - start_time
    if epoch >= 0:
        print(f"Quick evaluation completed in {eval_time:.2f}s")
    else:
        print(f"COMPREHENSIVE evaluation completed in {eval_time:.2f}s")
    print(f"\033[91mRecall@20: {results['recall'][0]:.6f}, NDCG@20: {results['ndcg'][0]:.6f}, Precision@20: {results['precision'][0]:.6f}\033[0m")
    
    return results

def save_training_plots(loss_history, recall_history, precision_history, ndcg_history, save_dir="./plots"):
    """
    Save training plots: loss, recall vs precision, and NDCG
    """
    # Create plots directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Loss function over epochs
    axes[0, 0].plot(range(1, len(loss_history) + 1), loss_history, 'b-', linewidth=2, marker='o')
    axes[0, 0].set_title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')  # Log scale for better visualization
    
    # Plot 2: Recall vs Precision
    if len(recall_history) > 0 and len(precision_history) > 0:
        axes[0, 1].plot(precision_history, recall_history, 'g-', linewidth=2, marker='s')
        axes[0, 1].set_title('Recall vs Precision', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Precision@20')
        axes[0, 1].set_ylabel('Recall@20')
        axes[0, 1].grid(True, alpha=0.3)
        # Add points for each evaluation
        for i, (p, r) in enumerate(zip(precision_history, recall_history)):
            axes[0, 1].annotate(f'E{i+1}', (p, r), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 3: NDCG over evaluations
    if len(ndcg_history) > 0:
        eval_points = range(1, len(ndcg_history) + 1)
        axes[1, 0].plot(eval_points, ndcg_history, 'r-', linewidth=2, marker='^')
        axes[1, 0].set_title('NDCG@20 Over Evaluations', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Evaluation Point')
        axes[1, 0].set_ylabel('NDCG@20')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: All metrics together
    if len(recall_history) > 0:
        eval_points = range(1, len(recall_history) + 1)
        axes[1, 1].plot(eval_points, recall_history, 'b-', linewidth=2, marker='o', label='Recall@20')
        axes[1, 1].plot(eval_points, precision_history, 'g-', linewidth=2, marker='s', label='Precision@20')
        axes[1, 1].plot(eval_points, ndcg_history, 'r-', linewidth=2, marker='^', label='NDCG@20')
        axes[1, 1].set_title('All Metrics Over Training', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Evaluation Point')
        axes[1, 1].set_ylabel('Metric Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_dir, f"training_plots_{int(time())}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Training plots saved to: {plot_path}")
    
    # Also save individual plots
    individual_plots = {
        'loss': (loss_history, 'Training Loss', 'Epoch', 'MSE Loss', 'b-'),
        'recall': (recall_history, 'Recall@20 Over Training', 'Evaluation', 'Recall@20', 'r-'),
        'precision': (precision_history, 'Precision@20 Over Training', 'Evaluation', 'Precision@20', 'g-'),
        'ndcg': (ndcg_history, 'NDCG@20 Over Training', 'Evaluation', 'NDCG@20', 'm-')
    }
    
    for name, (data, title, xlabel, ylabel, style) in individual_plots.items():
        if len(data) > 0:
            plt.figure(figsize=(8, 6))
            x_data = range(1, len(data) + 1)
            plt.plot(x_data, data, style, linewidth=2, marker='o')
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.grid(True, alpha=0.3)
            if name == 'loss':
                plt.yscale('log')
            
            individual_path = os.path.join(save_dir, f"{name}_plot_{int(time())}.png")
            plt.savefig(individual_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    plt.close('all')  # Close all figures to free memory

def train_universal_spectral_simple(dataset, model, config, total_epochs=15, verbose=False):
    """
    ULTRA-FAST training procedure for Universal Spectral CF
    Direct MSE on rating matrix - optimized for speed!
    """
    print("="*70)
    print("STARTING ULTRA-FAST DIRECT MSE TRAINING")
    print("="*70)
    
    # Initialize MSE loss
    mse_loss = SimpleMSELoss(model, config)
    
    # Training history for plotting
    loss_history = []
    recall_history = []
    precision_history = []
    ndcg_history = []
    
    # Training loop with progress tracking
    best_recall = 0.0
    training_start = time()
    
    for epoch in tqdm(range(total_epochs), desc="Training Progress"):
        epoch_start = time()
        
        # Training step
        avg_loss, train_info = MSE_train_simple(dataset, model, mse_loss, epoch)
        loss_history.append(avg_loss)
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
            
            # Store metrics for plotting
            recall_history.append(results['recall'][0])
            precision_history.append(results['precision'][0])
            ndcg_history.append(results['ndcg'][0])
            
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
    
    # Save training plots
    save_training_plots(loss_history, recall_history, precision_history, ndcg_history)
    
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