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
Recmodel = ClosedFormSpectralCF(adj_mat, world.config)

print(f"Device: {world.device}")
print("Initialized ClosedFormSpectralCF")

# # Test 1: Item spectral only
# cprint("[TEST 1: ITEM SPECTRAL ONLY]")

# def item_spectral_only_getUsersRating(batch_users):
#     with torch.no_grad():
#         user_profiles = Recmodel.adj_mat[batch_users]
#         item_scores = user_profiles @ Recmodel.item_filter_matrix
#         # 100% item spectral
#         return item_scores.cpu().numpy()

# Recmodel.getUsersRating = item_spectral_only_getUsersRating
# results_item = Procedure.Test(dataset, Recmodel, 0, world.config['multicore'])

# # Test 2: User spectral only
# cprint("[TEST 2: USER SPECTRAL ONLY]")

# def user_spectral_only_getUsersRating(batch_users):
#     with torch.no_grad():
#         user_filter_rows = Recmodel.user_filter_matrix[batch_users]
#         user_scores = user_filter_rows @ Recmodel.adj_mat
#         # 100% user spectral
#         return user_scores.cpu().numpy()

# Recmodel.getUsersRating = user_spectral_only_getUsersRating
# results_user = Procedure.Test(dataset, Recmodel, 0, world.config['multicore'])

# Test 3: Combined spectral (50% each)
cprint("[TEST 3: COMBINED SPECTRAL (50% ITEM + 50% USER)]")

def combined_spectral_getUsersRating(batch_users):
    with torch.no_grad():
        user_profiles = Recmodel.adj_mat[batch_users]
        item_scores = user_profiles @ Recmodel.item_filter_matrix
        user_filter_rows = Recmodel.user_filter_matrix[batch_users]
        user_scores = user_filter_rows @ Recmodel.adj_mat
        # 50% item + 50% user spectral
        combined_scores = item_scores + user_scores
        return combined_scores.cpu().numpy()

Recmodel.getUsersRating = combined_spectral_getUsersRating
results_combined = Procedure.Test(dataset, Recmodel, 0, world.config['multicore'])

# Results comparison
print("\n" + "="*70)
print("              SPECTRAL COMPONENTS COMPARISON")
print("="*70)
#print(f"Item Spectral Only:  Recall@20={results_item['recall'][0]:.6f}, NDCG@20={results_item['ndcg'][0]:.6f}, Precision@20={results_item['precision'][0]:.6f}")
#print(f"User Spectral Only:  Recall@20={results_user['recall'][0]:.6f}, NDCG@20={results_user['ndcg'][0]:.6f}, Precision@20={results_user['precision'][0]:.6f}")
print(f"Combined (50/50):    Recall@20={results_combined['recall'][0]:.6f}, NDCG@20={results_combined['ndcg'][0]:.6f}, Precision@20={results_combined['precision'][0]:.6f}")
print("="*70)

# Analysis
#item_recall = results_item['recall'][0]
#user_recall = results_user['recall'][0]
combined_recall = results_combined['recall'][0]

print("\nANALYSIS:")
# if item_recall > user_recall:
#     improvement = (item_recall - user_recall) / user_recall * 100
#     print(f"✅ Item spectral is BETTER: {improvement:.1f}% higher recall than user spectral")
# else:
#     improvement = (user_recall - item_recall) / item_recall * 100
#     print(f"✅ User spectral is BETTER: {improvement:.1f}% higher recall than item spectral")

# if combined_recall > max(item_recall, user_recall):
#     print("✅ Combining both gives the BEST results")
# elif combined_recall > min(item_recall, user_recall):
#     print("⚠️  Combination is middle ground between the two")
# else:
#     print("❌ Combination is worse than both individual components")

# print(f"\nBest individual component: {'Item' if item_recall > user_recall else 'User'} spectral")
# print(f"Best overall approach: {'Combined' if combined_recall == max(item_recall, user_recall, combined_recall) else ('Item' if item_recall > user_recall else 'User')} spectral")