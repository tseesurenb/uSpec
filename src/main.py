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