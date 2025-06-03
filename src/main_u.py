import world
import utils
from world import cprint
import torch
import SimpleProcedure  # Import our new simplified procedure
from tqdm import tqdm
import time

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

# Initialize Universal Spectral CF model with optimized settings
from model import UniversalSpectralCF

print(f"Device: {world.device}")

# Optimized configuration for faster training with direct MSE
# config = world.config.copy()
# config['filter_order'] = 2      # Simple filters for speed
# config['n_eigen'] = 20          # Reduced eigenvalues for speed  
# config['lr'] = 0.01             # Higher learning rate for direct MSE
# config['epochs'] = 15           # Very few epochs needed with direct MSE

print("Creating Universal Spectral CF model...")
model_start = time.time()

# Get the adjacency matrix
adj_mat = dataset.UserItemNet.tolil()

# Create the model with proper config
Recmodel = UniversalSpectralCF(adj_mat, world.config)

# Initialize the model (precompute eigendecompositions)
Recmodel.train()

print(f"Model created in {time.time() - model_start:.2f}s")

print("Initialized UniversalSpectralCF with direct MSE training (no negative sampling)")

# Simple one-line training and evaluation
print("Starting direct MSE-based training...")
training_start = time.time()

# Use the simplified training procedure
trained_model, final_results = SimpleProcedure.simple_train_and_evaluate(
    dataset, Recmodel, world.config
)

total_time = time.time() - training_start

# Display final results
print("\n" + "="*70)
print(f"              FINAL RESULTS (SIMPLIFIED MSE TRAINING) - {world.config['dataset']}, filter - {world.config['filter']}")
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