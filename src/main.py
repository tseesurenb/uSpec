'''
Created on June 3, 2025
Pytorch Implementation of uSpec in
Batsuuri. Tse et al. uSpec: Universal Spectral Collaborative Filtering

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import world
import utils
from world import cprint
import torch
import procedure  # Import our new simplified procedure
from tqdm import tqdm
import time
import register
from register import dataset

if world.config['m_type'] == 'single':
    from model_single import UniversalSpectralCF
else:
    from model_double import UniversalSpectralCF

import warnings
warnings.filterwarnings("ignore", message="Can't initialize NVML")


# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
print(f"Device: {world.device}")
# ==============================

print("Creating Universal Spectral CF model...")
model_start = time.time()

# Get the adjacency matrix
adj_mat = dataset.UserItemNet.tolil()

# Create the model with proper config
Recmodel = UniversalSpectralCF(adj_mat, world.config)
# Initialize the model (precompute eigendecompositions)
Recmodel.train()
print(f"Model created in {time.time() - model_start:.2f}s")
print("Initialized UniversalSpectralCF with direct MSE training")

# Simple one-line training and evaluation
print("Starting direct MSE-based training...")
training_start = time.time()

# Use the simplified training procedure
trained_model, final_results = procedure.train_and_evaluate(
    dataset, Recmodel, world.config
)

total_time = time.time() - training_start

# Display final results
print("\n" + "="*70)
print(f"              FINAL RESULTS (SIMPLIFIED MSE TRAINING) - {world.config['dataset']}, filter - {world.config['filter']}, {world.config['m_type']}")
print("="*70)
print(f"\033[91mFinal Results: Recall@20={final_results['recall'][0]:.6f}, NDCG@20={final_results['ndcg'][0]:.6f}, Precision@20={final_results['precision'][0]:.6f}\033[0m")
print(f"Total experiment time: {total_time:.2f}s")
print("="*70)

# Show what the model learned
print("\n[FINAL FILTER ANALYSIS]")
trained_model.debug_filter_learning()

print("\n🎉 Simplified Universal Spectral CF training completed!")
print("Key advantages of this approach:")
print("✅ MSE loss is much simpler than BPR")
print("✅ Faster convergence with fewer epochs")
print("✅ More stable gradients")
print("✅ Easier to tune hyperparameters")
print("✅ Direct optimization of rating prediction")