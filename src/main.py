'''
Created on June 3, 2025
PyTorch Implementation of uSpec: Universal Spectral Collaborative Filtering

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

# Create model
print(f"Creating Universal Spectral CF model (seed: {world.seed}, device: {world.device})...")
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

# Train and evaluate
print(f"\nStarting training with proper data splits...")
training_start = time.time()
trained_model, final_results = procedure.train_and_evaluate(dataset, Recmodel, world.config)
total_time = time.time() - training_start

# Display final results
print("\n" + "="*70)
print(f"FINAL RESULTS - {world.config['dataset']}, filter={world.config['filter']}, type={world.config['m_type']}")
print("="*70)
print(f"\033[91mFinal Test Results: Recall@20={final_results['recall'][0]:.6f}, NDCG@20={final_results['ndcg'][0]:.6f}, Precision@20={final_results['precision'][0]:.6f}\033[0m")
print(f"Total experiment time: {total_time:.2f}s")
print("="*70)

# Show learned filter patterns
print(f"\n[FINAL FILTER ANALYSIS - {world.config['m_type'].upper()}]")
trained_model.debug_filter_learning()

print(f"\n🎉 Universal Spectral CF training completed!")
print(f"📊 Model learned spectral coefficients using {dataset.trainDataSize:,} training interactions")
print(f"🎯 Best model selected using {dataset.valDataSize:,} validation interactions") 
print(f"🏆 Final evaluation on {len(dataset.testDict):,} test users")