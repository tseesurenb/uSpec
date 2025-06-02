import world
import utils
from world import cprint
import torch
import Procedure

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

# Initialize closed-form spectral model
from model import ClosedFormSpectralCF

adj_mat = dataset.UserItemNet.tolil()
config = {'device': world.device}
Recmodel = ClosedFormSpectralCF(adj_mat, config)

print(f"Device: {world.device}")
print("Initialized ClosedFormSpectralCF")

# Test 2: User spectral only
cprint("[TEST 2: USER SPECTRAL ONLY]")

def user_spectral_only_getUsersRating(batch_users):
    with torch.no_grad():
        user_filter_rows = Recmodel.user_filter_matrix[batch_users]
        user_scores = user_filter_rows @ Recmodel.adj_mat
        # 100% user spectral
        return user_scores.cpu().numpy()

Recmodel.getUsersRating = user_spectral_only_getUsersRating
results_user = Procedure.Test(dataset, Recmodel, 0, world.config['multicore'])

# Results comparison
print("\n" + "="*70)
print("              SPECTRAL COMPONENTS COMPARISON")
print("="*70)
print(f"User Spectral Only:  Recall@20={results_user['recall'][0]:.6f}, NDCG@20={results_user['ndcg'][0]:.6f}, Precision@20={results_user['precision'][0]:.6f}")
print("="*70)

