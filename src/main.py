'''
Created on June 3, 2025
PyTorch Implementation of uSpec: Universal Spectral Collaborative Filtering
Main script with filter design selection

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import world
import utils
import procedure
import time
from register import dataset
import warnings
warnings.filterwarnings("ignore", message="Can't initialize NVML")

import numpy as np

# Import appropriate model based on configuration
if world.config['m_type'] == 'single':
    from model_single import UniversalSpectralCF
else:
    from model_double import UniversalSpectralCF

# Set random seed for reproducibility
utils.set_seed(world.seed)

# Display configuration
print(f"Universal Spectral CF Configuration:")
print(f"  └─ Filter Design: {world.config.get('filter_design', 'original')}")
print(f"  └─ Initialization: {world.config.get('init_filter', 'smooth')}")
print(f"  └─ Filter Type: {world.config['filter']}")
print(f"  └─ Filter Order: {world.config['filter_order']}")
print(f"  └─ Model Type: {world.config['m_type']}")

# Create model
print(f"\nCreating Universal Spectral CF model (seed: {world.seed}, device: {world.device})...")
model_start = time.time()
adj_mat = dataset.UserItemNet.tolil()  # Use training data only
Recmodel = UniversalSpectralCF(adj_mat, world.config)
print(f"Model created in {time.time() - model_start:.2f}s")

# Display dataset information
print(f"\nDataset Information:")
print(f"  └─ Dataset: {world.config['dataset']}")
print(f"  └─ Users: {dataset.n_users:,}")
print(f"  └─ Items: {dataset.m_items:,}")
print(f"  └─ Training interactions: {dataset.trainDataSize:,}")
print(f"  └─ Validation interactions: {dataset.valDataSize:,}")
print(f"  └─ Test users: {len(dataset.testDict):,}")

# Check validation split
if dataset.valDataSize > 0:
    print(f"✅ Proper train/validation/test split detected")
    print(f"   Training will use validation data for model selection")
else:
    print(f"⚠️  No validation split - will use test data during training")

# Check if convergence test is requested
if world.config.get('run_convergence_test', False):
    print(f"\n🧪 Running comprehensive convergence test...")
    convergence_results = procedure.run_convergence_test(dataset, world.config)
else:
    # Standard training
    print(f"\nStarting training...")
    training_start = time.time()
    trained_model, final_results = procedure.train_and_evaluate(dataset, Recmodel, world.config)
    total_time = time.time() - training_start

    # Display final results
    print("\n" + "="*70)
    print(f"FINAL RESULTS - {world.config['dataset']}")
    print(f"Filter: {world.config['filter']}, Design: {world.config.get('filter_design', 'original')}")
    print(f"Init: {world.config.get('init_filter', 'smooth')}, Type: {world.config['m_type']}")
    print("="*70)
    print(f"\033[91mFinal Test Results: Recall@20={final_results['recall'][0]:.6f}, NDCG@20={final_results['ndcg'][0]:.6f}, Precision@20={final_results['precision'][0]:.6f}\033[0m")
    print(f"Total experiment time: {total_time:.2f}s")
    print("="*70)

    # Show learned filter patterns
    print(f"\n[FINAL FILTER ANALYSIS - {world.config.get('filter_design', 'original').upper()}]")
    trained_model.debug_filter_learning()

    print(f"\n🎉 Universal Spectral CF training completed!")
    print(f"📊 Model learned spectral coefficients using {dataset.trainDataSize:,} training interactions")
    print(f"🎯 Best model selected using {dataset.valDataSize:,} validation interactions") 
    print(f"🏆 Final evaluation on {len(dataset.testDict):,} test users")

# Usage examples and quick tests
def quick_comparison_test():
    """Quick test to compare different filter designs with same initialization"""
    print(f"\n{'='*60}")
    print("🚀 QUICK COMPARISON TEST")
    print(f"{'='*60}")
    
    designs = ['original', 'basis']
    init_filter = 'smooth'
    
    results = {}
    
    for design in designs:
        print(f"\n🔧 Testing {design} filter with {init_filter} initialization...")
        
        config_copy = world.config.copy()
        config_copy['filter_design'] = design
        config_copy['init_filter'] = init_filter
        config_copy['epochs'] = 50  # Shorter for quick test
        config_copy['patience'] = 8
        
        # Create and train model
        adj_mat = dataset.UserItemNet.tolil()
        model = UniversalSpectralCF(adj_mat, config_copy)
        trained_model, final_results = procedure.train_and_evaluate(dataset, model, config_copy)
        
        results[design] = final_results['ndcg'][0]
        print(f"   Result: NDCG@20 = {final_results['ndcg'][0]:.6f}")
    
    print(f"\n📊 Comparison Results:")
    for design, ndcg in results.items():
        print(f"   {design:10}: {ndcg:.6f}")
    
    best_design = max(results.keys(), key=lambda k: results[k])
    improvement = results[best_design] - min(results.values())
    print(f"\n🏆 Best: {best_design} (improvement: +{improvement:.6f})")

def initialization_test():
    """Test same filter design with different initializations"""
    print(f"\n{'='*60}")
    print("🎯 INITIALIZATION ROBUSTNESS TEST")
    print(f"{'='*60}")
    
    design = 'basis'  # Use basis filter for this test
    initializations = ['smooth', 'golden_036', 'butterworth']
    
    results = {}
    
    for init in initializations:
        print(f"\n🔧 Testing {design} filter with {init} initialization...")
        
        config_copy = world.config.copy()
        config_copy['filter_design'] = design
        config_copy['init_filter'] = init
        config_copy['epochs'] = 50
        config_copy['patience'] = 8
        
        # Create and train model
        adj_mat = dataset.UserItemNet.tolil()
        model = UniversalSpectralCF(adj_mat, config_copy)
        trained_model, final_results = procedure.train_and_evaluate(dataset, model, config_copy)
        
        results[init] = final_results['ndcg'][0]
        print(f"   Result: NDCG@20 = {final_results['ndcg'][0]:.6f}")
    
    print(f"\n📊 Initialization Results:")
    ndcgs = list(results.values())
    for init, ndcg in results.items():
        print(f"   {init:12}: {ndcg:.6f}")
    
    gap = max(ndcgs) - min(ndcgs)
    std_dev = np.std(ndcgs)
    print(f"\n📈 Convergence Analysis:")
    print(f"   Gap:     {gap:.6f}")
    print(f"   Std Dev: {std_dev:.6f}")
    
    if gap < 0.01:
        print(f"   ✅ Excellent convergence!")
    elif gap < 0.02:
        print(f"   🟢 Good convergence")
    else:
        print(f"   🟡 Room for improvement")

# Uncomment to run quick tests
# quick_comparison_test()
# initialization_test()