'''
Created on June 3, 2025
Pytorch Implementation of uSpec in
Batsuuri. Tse et al. uSpec: Universal Spectral Collaborative Filtering

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
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

class MSELoss:
    def __init__(self, model, config):
        self.model = model
        self.lr = config['lr']
        self.weight_decay = config.get('decay', 1e-4)
        
        self.opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
    def compute_loss(self, users, target_ratings):
        
        predicted_ratings = self.model(users, target_ratings=None)
        
        if isinstance(predicted_ratings, np.ndarray):
            predicted_ratings = torch.from_numpy(predicted_ratings).to(world.device)
        
        mse_loss = torch.mean((predicted_ratings - target_ratings) ** 2)
        reg_loss = 0.0
        if hasattr(self.model, 'user_filter') and hasattr(self.model.user_filter, 'coeffs'):
            reg_loss += self.model.user_filter.coeffs.norm(2).pow(2) * 1e-6
        if hasattr(self.model, 'item_filter') and hasattr(self.model.item_filter, 'coeffs'):
            reg_loss += self.model.item_filter.coeffs.norm(2).pow(2) * 1e-6
        
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
    batch_size = len(users)
    n_items = dataset.m_items
    
    target_ratings = torch.zeros(batch_size, n_items)
    
    for i, user in enumerate(users):
        pos_items = dataset.allPos[user]
        if len(pos_items) > 0:
            target_ratings[i, pos_items] = 1.0
    
    return target_ratings

def train(dataset, model, loss_class, epoch):
    model.train()

    mse_loss = loss_class

    n_users = dataset.n_users
    batch_size = 512  # Increased batch size for better coverage
    
    # FIXED: Better sampling strategy - ensure all users are seen over time
    users_per_epoch = min(n_users, max(2000, n_users // 3))  # Increased coverage
    n_batch = users_per_epoch // batch_size
    
    total_loss = 0.0
    start_time = time()
    
    # FIXED: Use random sampling with better coverage
    # Sample different users each epoch to ensure full dataset coverage
    np.random.seed(epoch * 42)  # Different seed each epoch for variety
    sampled_users = np.random.choice(n_users, users_per_epoch, replace=False)
    
    for batch_idx in range(n_batch):
        # Get batch of users
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, users_per_epoch)
        
        user_indices = sampled_users[start_idx:end_idx]
        
        users = torch.LongTensor(user_indices).to(world.device)
        
        # Create target ratings
        target_ratings = create_target_ratings(dataset, user_indices)
        target_ratings = target_ratings.to(world.device)
        
        # Training step
        batch_loss = mse_loss.train_step(users, target_ratings)
        total_loss += batch_loss
    
    # Ensure we have at least one batch
    if n_batch == 0:
        n_batch = 1
        users = torch.LongTensor(np.random.choice(n_users, min(batch_size, n_users), replace=False)).to(world.device)
        target_ratings = create_target_ratings(dataset, users.cpu().numpy())
        target_ratings = target_ratings.to(world.device)
        batch_loss = mse_loss.train_step(users, target_ratings)
        total_loss = batch_loss
    
    avg_loss = total_loss / n_batch
    training_time = time() - start_time
    
    return avg_loss, f"MSE_loss: {avg_loss:.4f} | Batches: {n_batch} | Users/epoch: {users_per_epoch} | Coverage: {100*users_per_epoch/n_users:.1f}% | Time: {training_time:.2f}s"

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

def test(dataset, model, epoch, multicore=0):
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
        
        # FIXED: Better sampling strategy for quick evaluation
        if epoch >= 0:  # During training - FAST but REPRESENTATIVE
            # Use stratified sampling instead of just first 200 users
            num_quick_users = min(len(users), 500)  # Increased from 200 to 500
            
            # Group users by activity level for stratified sampling
            user_activity = [(u, len(dataset.allPos[u])) for u in users]
            user_activity.sort(key=lambda x: x[1])  # Sort by activity
            
            # Sample from different activity levels
            n_strata = 5
            strata_size = len(user_activity) // n_strata
            test_users = []
            
            users_per_stratum = num_quick_users // n_strata
            for i in range(n_strata):
                start_idx = i * strata_size
                end_idx = (i + 1) * strata_size if i < n_strata - 1 else len(user_activity)
                stratum_users = [u for u, _ in user_activity[start_idx:end_idx]]
                
                # Randomly sample from this stratum
                if len(stratum_users) > users_per_stratum:
                    sampled = np.random.choice(stratum_users, users_per_stratum, replace=False)
                else:
                    sampled = stratum_users
                test_users.extend(sampled)
            
            # Fill remaining slots with random users if needed
            while len(test_users) < num_quick_users and len(test_users) < len(users):
                remaining_users = [u for u in users if u not in test_users]
                if remaining_users:
                    test_users.append(np.random.choice(remaining_users))
                else:
                    break
                    
            batch_size = min(u_batch_size * 2, 200)  # Larger batches for speed
            print(f"Quick eval on {len(test_users)} users (stratified sample)...")
            
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

def train_universal_spectral(dataset, model, config, total_epochs=15, verbose=1):
    """
    FIXED: Better evaluation schedule and monitoring
    """
    print("="*70)
    print("STARTING ULTRA-FAST DIRECT MSE TRAINING")
    print("="*70)
    
    # Initialize MSE loss
    mse_loss = MSELoss(model, config)
    
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
        avg_loss, train_info = train(dataset, model, mse_loss, epoch)
        loss_history.append(avg_loss)
        epoch_time = time() - epoch_start
        
        # Print progress every 5 epochs only
        if verbose == 1:
            if epoch % 5 == 0 or epoch == total_epochs - 1:
                print(f"\nEpoch {epoch+1}/{total_epochs}: {train_info}")
                model.debug_filter_learning()
        
        # FIXED: More frequent evaluation to track training progress
        n_epoch_eval = config.get('n_epoch_eval', 5)  # Default every 5 epochs
        if (epoch + 1) % n_epoch_eval == 0 or epoch == total_epochs - 1:
        #if epoch == total_epochs // 3 or epoch == 2 * total_epochs // 3 or epoch == total_epochs - 1:
            print(f"\n[QUICK EVAL - Epoch {epoch+1}]")
            results = test(dataset, model, epoch, world.config['multicore'])
            
            # Store metrics for plotting
            recall_history.append(results['recall'][0])
            precision_history.append(results['precision'][0])
            ndcg_history.append(results['ndcg'][0])
            
            # Track best performance
            current_recall = results['recall'][0]
            if current_recall > best_recall:
                best_recall = current_recall
                print(f"✅ New best recall: {best_recall:.6f}")
            
            # Early stopping if performance degrades significantly
            if len(recall_history) > 1 and current_recall < 0.7 * max(recall_history):
                print(f"⚠️  Performance dropped significantly, consider stopping early")
    
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
def train_and_evaluate(dataset, model, config):
    """
    One-line training and evaluation for main_u.py
    """
    # Training with the epochs from config
    trained_model = train_universal_spectral(
        dataset, model, config, 
        total_epochs=config.get('epochs', 15),  # Use config epochs
        verbose=config.get('verbose', 1)  # Use config verbosity
    )
    
    # Final comprehensive evaluation
    print("\n[FINAL COMPREHENSIVE EVALUATION - ALL USERS, ALL ITEMS]")
    final_results = test(dataset, trained_model, -1, world.config['multicore'])
    
    return trained_model, final_results


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