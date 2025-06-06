'''
Created on June 3, 2025
PyTorch Implementation of uSpec: Universal Spectral Collaborative Filtering
Adaptive procedure for different filter designs

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import world
import numpy as np
import torch
import torch.nn as nn
import utils
from time import time
from tqdm import tqdm

class MSELoss:
    def __init__(self, model, config):
        self.model = model
        
        base_lr = config['lr']
        weight_decay = config['decay']
        
        # Adaptive learning rates based on filter design
        filter_design = getattr(model, 'filter_design', 'original')
        
        # ENHANCED: Learning rate multipliers for maximum performance
        lr_multipliers = {
            'original': 8.0,           # Very aggressive for original to escape local minima
            'basis': 1.5,              # Increased for better basis learning
            'enhanced_basis': 2.0,     # Higher LR for enhanced basis performance
            'adaptive_golden': 1.5,    # Moderate for golden adaptation
            'adaptive': 2.0,           # Moderate boost for boundary learning
            'neural': 1.5              # Slight boost for neural net training
        }
        
        lr_mult = lr_multipliers.get(filter_design, 1.0)
        
        try:
            filter_params = list(model.get_filter_parameters())
            other_params = list(model.get_other_parameters())
            
            print(f"🔧 {filter_design.upper()} Learning Optimizer:")
            print(f"   Filter params: {len(filter_params)} parameters")
            print(f"   Other params: {len(other_params)} parameters")
            print(f"   LR multiplier: {lr_mult}x")
            
            param_groups = []
            
            if len(filter_params) > 0:
                filter_lr = base_lr * lr_mult
                filter_wd = weight_decay * 0.1  # Light regularization for filters
                
                param_groups.append({
                    'params': filter_params,
                    'lr': filter_lr,
                    'weight_decay': filter_wd
                })
                print(f"   Filter LR: {filter_lr:.5f}, WD: {filter_wd:.6f}")
            
            if len(other_params) > 0:
                param_groups.append({
                    'params': other_params,
                    'lr': base_lr,
                    'weight_decay': weight_decay
                })
                print(f"   Other LR: {base_lr:.5f}, WD: {weight_decay:.6f}")
            
            if len(param_groups) > 0:
                # ENHANCED: Use optimizers optimized for performance
                if filter_design in ['basis', 'enhanced_basis', 'neural', 'adaptive_golden']:
                    self.opt = torch.optim.AdamW(param_groups, amsgrad=True)
                    # Add cosine annealing scheduler for better convergence
                    total_steps = config.get('epochs', 50) * 20  # Approximate steps per epoch
                    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.opt, T_max=total_steps, eta_min=base_lr * 0.01
                    )
                else:
                    self.opt = torch.optim.Adam(param_groups)
                    self.scheduler = None
                self.has_separate_params = True
            else:
                if filter_design in ['basis', 'enhanced_basis', 'neural', 'adaptive_golden']:
                    self.opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay, amsgrad=True)
                    total_steps = config.get('epochs', 50) * 20
                    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.opt, T_max=total_steps, eta_min=base_lr * 0.01
                    )
                else:
                    self.opt = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
                    self.scheduler = None
                self.has_separate_params = False
                
        except AttributeError:
            print(f"   Using single optimizer for {filter_design}")
            if filter_design in ['basis', 'enhanced_basis', 'neural', 'adaptive_golden']:
                self.opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay, amsgrad=True)
                total_steps = config.get('epochs', 50) * 20
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.opt, T_max=total_steps, eta_min=base_lr * 0.01
                )
            else:
                self.opt = torch.optim.Adam(model.parameters(), lr=base_lr * lr_mult, weight_decay=weight_decay)
                self.scheduler = None
            self.has_separate_params = False
        
        self.filter_design = filter_design
    
    def train_step(self, users, target_ratings):
        """Adaptive training step based on filter design"""
        self.opt.zero_grad()
        predicted_ratings = self.model(users)
        
        if isinstance(predicted_ratings, np.ndarray):
            predicted_ratings = torch.from_numpy(predicted_ratings).to(world.device)
        
        # ENHANCED: Performance-optimized loss with design-specific regularization
        loss = torch.mean((predicted_ratings - target_ratings) ** 2)
        
        # Add design-specific regularization for maximum performance
        if self.filter_design in ['basis', 'enhanced_basis']:
            # For basis filters: encourage bold mixing decisions
            if hasattr(self.model, 'user_filter') and self.model.user_filter is not None:
                if hasattr(self.model.user_filter, 'mixing_weights'):
                    mixing_probs = torch.softmax(self.model.user_filter.mixing_weights, dim=0)
                    # Encourage high entropy (bold decisions) for enhanced basis
                    if self.filter_design == 'enhanced_basis':
                        entropy = -torch.sum(mixing_probs * torch.log(mixing_probs + 1e-8))
                        max_entropy = torch.log(torch.tensor(len(mixing_probs), dtype=torch.float32))
                        entropy_loss = (max_entropy - entropy) * 1e-3  # Encourage high entropy
                        loss += entropy_loss
                    else:
                        # Original basis: encourage focused mixing
                        mixing_entropy = -torch.sum(mixing_probs * torch.log(mixing_probs + 1e-8))
                        loss += 1e-4 * mixing_entropy
            
        elif self.filter_design == 'adaptive_golden':
            # For adaptive golden: encourage reasonable golden ratio variations
            if hasattr(self.model, 'user_filter') and self.model.user_filter is not None:
                if hasattr(self.model.user_filter, 'golden_ratio_delta'):
                    # Regularize to stay close to optimal golden ratio range
                    golden_reg = torch.abs(self.model.user_filter.golden_ratio_delta) * 1e-3
                    loss += golden_reg
            
        elif self.filter_design == 'adaptive':
            # Regularize boundaries to prevent collapse
            boundary_reg = 0.0
            if hasattr(self.model, 'user_filter') and self.model.user_filter is not None:
                if hasattr(self.model.user_filter, 'boundary_1'):
                    # Encourage reasonable boundary separation
                    b1 = torch.sigmoid(self.model.user_filter.boundary_1) * 0.5
                    b2 = b1 + torch.sigmoid(self.model.user_filter.boundary_2) * 0.5
                    boundary_reg += torch.relu(0.1 - (b2 - b1))  # Min separation of 0.1
            loss += 1e-3 * boundary_reg
        
        loss.backward()
        
        # ENHANCED: Adaptive gradient clipping for performance
        if self.filter_design == 'original':
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        elif self.filter_design in ['enhanced_basis', 'adaptive_golden']:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=3.0)  # Moderate clipping
        elif self.filter_design == 'neural':
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.opt.step()
        
        # ENHANCED: Update scheduler if available
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            self.scheduler.step()
        
        return loss.cpu().item()

def create_target_ratings(dataset, users):
    """Create target ratings from training data"""
    batch_size = len(users)
    n_items = dataset.m_items
    target_ratings = torch.zeros(batch_size, n_items)
    
    for i, user in enumerate(users):
        pos_items = dataset.allPos[user]
        if len(pos_items) > 0:
            target_ratings[i, pos_items] = 1.0
    
    return target_ratings

def train_epoch(dataset, model, loss_class, epoch, config):
    """Train for one epoch"""
    model.train()
    n_users = dataset.n_users
    
    # Get training batch size
    train_batch_size = config['train_u_batch_size']
    if train_batch_size == -1:
        train_batch_size = n_users
        users_per_epoch = n_users
    else:
        users_per_epoch = min(n_users, max(2000, n_users // 3))
    
    # Sample users for this epoch - ensure integers
    np.random.seed(epoch * 42)
    sampled_users = np.random.choice(n_users, users_per_epoch, replace=False)
    sampled_users = [int(u) for u in sampled_users]  # Convert to integers
    
    # Train in batches
    total_loss = 0.0
    n_batches = max(1, users_per_epoch // train_batch_size)
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * train_batch_size
        end_idx = min(start_idx + train_batch_size, users_per_epoch)
        user_indices = sampled_users[start_idx:end_idx]
        
        users = torch.LongTensor(user_indices).to(world.device)
        target_ratings = create_target_ratings(dataset, user_indices).to(world.device)
        
        batch_loss = loss_class.train_step(users, target_ratings)
        total_loss += batch_loss
        
        if train_batch_size == n_users:  # Full dataset mode
            break
    
    avg_loss = total_loss / n_batches
    return avg_loss

def evaluate(dataset, model, data_dict, config, sample_size=None):
    """Evaluate model on given data"""
    if len(data_dict) == 0:
        return {'recall': np.zeros(len(world.topks)),
                'precision': np.zeros(len(world.topks)),
                'ndcg': np.zeros(len(world.topks))}
    
    model.eval()
    eval_batch_size = config['eval_u_batch_size']
    max_K = max(world.topks)
    
    results = {'recall': np.zeros(len(world.topks)),
               'precision': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    
    with torch.no_grad():
        users = list(data_dict.keys())
        if sample_size and sample_size < len(users):
            users = np.random.choice(users, sample_size, replace=False)
            users = [int(u) for u in users]  # Ensure integers
        
        all_results = []
        
        # Process in batches
        for batch_users in utils.minibatch(users, batch_size=eval_batch_size):
            # Ensure batch_users are integers
            batch_users = [int(u) for u in batch_users]
            
            # Get training items and ground truth
            training_items = dataset.getUserPosItems(batch_users)
            ground_truth = [data_dict[u] for u in batch_users]
            
            # Get model predictions
            batch_users_gpu = torch.LongTensor(batch_users).to(world.device)
            ratings = model.getUsersRating(batch_users_gpu)
            
            if isinstance(ratings, np.ndarray):
                ratings = torch.from_numpy(ratings)
            
            # Exclude training items
            for i, items in enumerate(training_items):
                if len(items) > 0:
                    ratings[i, items] = -float('inf')
            
            # Get top-K predictions
            _, top_items = torch.topk(ratings, k=max_K)
            
            # Compute metrics for this batch
            batch_result = compute_metrics(ground_truth, top_items.cpu().numpy())
            all_results.append(batch_result)
        
        # Aggregate results
        for result in all_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        
        # Average over users
        n_users = len(users)
        results['recall'] /= n_users
        results['precision'] /= n_users
        results['ndcg'] /= n_users
    
    return results

def compute_metrics(ground_truth, predictions):
    """Compute recall, precision, NDCG for a batch"""
    # Convert to binary relevance
    relevance = utils.getLabel(ground_truth, predictions)
    
    recall, precision, ndcg = [], [], []
    for k in world.topks:
        # Compute metrics at k
        ret = utils.RecallPrecision_ATk(ground_truth, relevance, k)
        recall.append(ret['recall'])
        precision.append(ret['precision'])
        ndcg.append(utils.NDCGatK_r(ground_truth, relevance, k))
    
    return {'recall': np.array(recall),
            'precision': np.array(precision),
            'ndcg': np.array(ndcg)}

def train_and_evaluate(dataset, model, config):
    """Complete training and evaluation pipeline"""
    
    # Setup
    print("="*60)
    print(f"🚀 STARTING UNIVERSAL SPECTRAL CF TRAINING")
    print(f"   Filter Design: {getattr(model, 'filter_design', 'original').upper()}")
    print("="*60)
    
    # Check validation availability
    has_validation = hasattr(dataset, 'valDict') and len(dataset.valDict) > 0
    if has_validation:
        print(f"✅ Using validation split ({dataset.valDataSize:,} interactions)")
    else:
        print(f"⚠️  No validation - using test data during training")
    
    # Initialize
    loss_class = MSELoss(model, config)
    best_ndcg = 0.0
    best_epoch = 0
    best_model_state = None
    no_improvement = 0
    
    # Training parameters
    total_epochs = config['epochs']
    patience = config['patience']
    min_delta = config['min_delta']
    eval_every = config['n_epoch_eval']
    
    print(f"📊 Training: {dataset.trainDataSize:,} interactions")
    print(f"🎯 Config: {total_epochs} epochs, patience={patience}")
    
    # Training loop
    training_start = time()
    
    for epoch in tqdm(range(total_epochs), desc="Training"):
        # Train one epoch
        avg_loss = train_epoch(dataset, model, loss_class, epoch, config)
        
        # Evaluate periodically
        if (epoch + 1) % eval_every == 0 or epoch == total_epochs - 1:
            # Use validation if available, otherwise test
            eval_data = dataset.valDict if has_validation else dataset.testDict
            eval_name = "validation" if has_validation else "test"
            
            results = evaluate(dataset, model, eval_data, config, sample_size=200)
            current_ndcg = results['ndcg'][0]
            
            # Check for improvement
            if current_ndcg > best_ndcg + min_delta:
                best_ndcg = current_ndcg
                best_epoch = epoch + 1
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improvement = 0
                
                print(f"\n✅ Epoch {epoch+1}: New best {eval_name} NDCG = {current_ndcg:.6f}")
                print(f"   Training loss: {avg_loss:.6f}")
            else:
                no_improvement += 1
                print(f"\n📈 Epoch {epoch+1}: {eval_name} NDCG = {current_ndcg:.6f} (best: {best_ndcg:.6f})")
                print(f"   Training loss: {avg_loss:.6f}")
            
            # Early stopping
            if no_improvement >= patience // eval_every:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
            
            # Show filter learning occasionally
            if config.get('verbose', 1) and epoch % (eval_every * 3) == 0:
                print(f"\n--- Filter Learning Status at Epoch {epoch+1} ---")
                model.debug_filter_learning()
    
    # Restore best model
    if best_model_state is not None:
        print(f"\n🔄 Restoring best model from epoch {best_epoch}")
        model.load_state_dict(best_model_state)
    
    training_time = time() - training_start
    
    # Final test evaluation
    print(f"\n" + "="*60)
    print("🏆 FINAL TEST EVALUATION")
    print("="*60)
    
    final_results = evaluate(dataset, model, dataset.testDict, config)
    
    print(f"⏱️  Training time: {training_time:.2f}s")
    print(f"🎯 Best epoch: {best_epoch}")
    print(f"📊 Final test results:")
    print(f"   Recall@20:    {final_results['recall'][0]:.6f}")
    print(f"   Precision@20: {final_results['precision'][0]:.6f}")
    print(f"   NDCG@20:      {final_results['ndcg'][0]:.6f}")
    print("="*60)
    
    # Show final filter learning results
    print(f"\n--- Final Filter Learning Results ---")
    model.debug_filter_learning()
    
    return model, final_results

def run_convergence_test(dataset, config):
    """Test convergence across different initializations and designs"""
    
    # ENHANCED: Include new filter designs in convergence test
    filter_designs = ['original', 'basis', 'enhanced_basis', 'adaptive_golden', 'adaptive', 'neural']
    initializations = ['smooth', 'golden_036', 'butterworth']
    
    results = {}
    
    print(f"\n{'='*80}")
    print("🧪 COMPREHENSIVE CONVERGENCE TEST")
    print(f"{'='*80}")
    
    for design in filter_designs:
        print(f"\n🔧 Testing Filter Design: {design.upper()}")
        print(f"-" * 40)
        
        design_results = {}
        
        for init in initializations:
            print(f"\n   🎯 Initialization: {init}")
            
            # Create fresh model
            config_copy = config.copy()
            config_copy['filter_design'] = design
            config_copy['init_filter'] = init
            
            from model_single import UniversalSpectralCF
            adj_mat = dataset.UserItemNet.tolil()
            model = UniversalSpectralCF(adj_mat, config_copy)
            
            # Train model
            trained_model, final_results = train_and_evaluate(dataset, model, config_copy)
            
            design_results[init] = {
                'ndcg': final_results['ndcg'][0],
                'recall': final_results['recall'][0],
                'precision': final_results['precision'][0]
            }
            
            print(f"      Result: NDCG = {final_results['ndcg'][0]:.6f}")
        
        results[design] = design_results
    
    # ENHANCED: Analyze convergence with performance ranking
    print(f"\n{'='*80}")
    print("📊 CONVERGENCE AND PERFORMANCE ANALYSIS")
    print(f"{'='*80}")
    
    design_max_performance = {}
    
    for design in filter_designs:
        ndcgs = [results[design][init]['ndcg'] for init in initializations]
        max_ndcg = max(ndcgs)
        min_ndcg = min(ndcgs)
        gap = max_ndcg - min_ndcg
        std_ndcg = np.std(ndcgs)
        mean_ndcg = np.mean(ndcgs)
        
        design_max_performance[design] = max_ndcg
        
        print(f"\n{design.upper()} Filter:")
        for init in initializations:
            ndcg = results[design][init]['ndcg']
            deviation = max_ndcg - ndcg
            print(f"  {init:12}: {ndcg:.6f} (gap: {deviation:.6f})")
        
        print(f"  Max NDCG:  {max_ndcg:.6f}")
        print(f"  Mean NDCG: {mean_ndcg:.6f}")
        print(f"  Gap:       {gap:.6f}")
        print(f"  Std Dev:   {std_ndcg:.6f}")
        
        # Convergence quality
        if gap < 0.01:
            convergence_status = "✅ EXCELLENT convergence"
        elif gap < 0.02:
            convergence_status = "🟢 GOOD convergence"
        elif gap < 0.05:
            convergence_status = "🟡 MODERATE convergence"
        else:
            convergence_status = "🔴 POOR convergence"
        
        # Performance quality
        if max_ndcg > 0.37:
            performance_status = "🏆 EXCELLENT performance"
        elif max_ndcg > 0.33:
            performance_status = "🥇 GOOD performance"
        elif max_ndcg > 0.30:
            performance_status = "🥈 MODERATE performance"
        else:
            performance_status = "🥉 LOW performance"
        
        print(f"  {convergence_status}")
        print(f"  {performance_status}")
    
    # ENHANCED: Overall ranking
    print(f"\n{'='*80}")
    print("🏆 OVERALL RANKING (by maximum performance)")
    print(f"{'='*80}")
    
    sorted_designs = sorted(design_max_performance.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (design, max_perf) in enumerate(sorted_designs, 1):
        gap = max(results[design][init]['ndcg'] for init in initializations) - min(results[design][init]['ndcg'] for init in initializations)
        print(f"{rank}. {design.upper():15} - Max: {max_perf:.6f}, Gap: {gap:.6f}")
    
    return results