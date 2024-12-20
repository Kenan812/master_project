import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import normalize
import ast
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------
# Setup GPU device (fallback CPU)
# -------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# -------------------
# Load Data
# -------------------
metadata = pd.read_csv('datasets/filtered_metadata.csv')
reviews = pd.read_csv('datasets/filtered_reviews.csv')
img_data = np.load('datasets/filtered_image_features.npz')

# -------------------
# Filter items with image embeddings
# -------------------
img_asins = set(img_data.keys())
metadata = metadata[metadata['parent_asin'].isin(img_asins)].reset_index(drop=True)
metadata_asins = metadata['parent_asin'].tolist()
print(f'Number of items with image embeddings: {len(metadata_asins)}')

# -------------------
# Create item embeddings using only image features
# -------------------
img_embs = np.array([img_data[a] for a in metadata_asins])
item_embs = normalize(img_embs, norm='l2', axis=1).astype(np.float32)
item_embs_tensor = torch.tensor(item_embs, dtype=torch.float32).to(device)
num_items = len(metadata_asins)
print(f"item_embs shape (only image features): {item_embs.shape}")

# -------------------
# Replace missing values in metadata
# -------------------
metadata['manufacturer_id'] = metadata['manufacturer_id'].fillna(-1).astype(int)
metadata['rated_id'] = metadata['rated_id'].fillna(-1).astype(int)
metadata['average_rating'] = metadata['average_rating'].fillna(0.0).astype(float)

category_cols = ['category_1_id', 'category_2_id', 'category_3_id', 'category_4_id', 'category_5_id']
for c in category_cols:
    metadata[c].fillna(-1, inplace=True)
    metadata[c] = metadata[c].astype(int)

# -------------------
# List of alpha and beta pairs to iterate over
# -------------------
alpha_beta_pairs = [
    (1, 0),
    (0.8, 0.2),
    (0.6, 0.4),
    (0.5, 0.5),
    (0.4, 0.6),
    (0.2, 0.8),
    (0, 1),
]

# -------------------
# Loop over each alpha and beta pair
# -------------------
for alpha, beta in alpha_beta_pairs:
    print(f"\nRunning with alpha={alpha}, beta={beta}")

    # -------------------
    # Prepare user and item indexing
    # -------------------
    user_ids = reviews['user_id'].unique().tolist()
    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    item_id_to_idx = {asin: i for i, asin in enumerate(metadata_asins)}

    num_users = len(user_ids)
    print(f'Number of reviews: {len(reviews)}')
    print(f'Number of metadata entries: {len(metadata)}')
    print(f'Number of users: {num_users}')
    print(f'Number of items: {num_items}')

    # -------------------
    # Create user-item edges with rating and sentiment features
    # -------------------
    edge_user = []
    edge_item = []
    edge_attr_mean = []

    for row in reviews.itertuples(index=False):
        if row.parent_asin not in item_id_to_idx:
            continue
        u = user_id_to_idx[row.user_id]
        i = item_id_to_idx[row.parent_asin]

        normalized_rating = (row.rating - 1.0) / 4.0
        text_s = (row.text_sentiment + 1) / 2.0
        title_s = (row.title_sentiment+1) / 2.0
        mean_edge_feat = (normalized_rating + text_s + title_s) / 3.0

        edge_user.append(u)
        edge_item.append(i)
        edge_attr_mean.append(mean_edge_feat)

    edge_user = torch.tensor(edge_user, dtype=torch.long)
    edge_item = torch.tensor(edge_item, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_mean, dtype=torch.float).unsqueeze(1)

    # -------------------
    # Construct HeteroData graph
    # -------------------
    data = HeteroData()
    data['user'].num_nodes = num_users
    data['item'].num_nodes = num_items

    if 'mapped_features' in metadata.columns:
        item_features_list = metadata['mapped_features'].apply(lambda x: set(ast.literal_eval(x)) if isinstance(x, str) else set()).tolist()
    else:
        item_features_list = [set() for _ in range(len(metadata))]

    data['user', 'reviews', 'item'].edge_index = torch.stack([edge_user, edge_item], dim=0)
    data['user', 'reviews', 'item'].edge_attr = edge_attr

    print("Heterogeneous graph constructed!")
    print(data)

    # -------------------
    # Prepare arrays for similarity logic
    # -------------------
    avg_ratings = metadata['average_rating'].fillna(0.0).values
    rated_ids   = metadata['rated_id'].fillna(-1).astype(int).values
    man_ids     = metadata['manufacturer_id'].fillna(-1).astype(int).values
    categories  = metadata[category_cols].fillna(-1).astype(int).values

    category_map = {
        i: set(cat for cat in categories[i] if cat != -1)
        for i in range(len(metadata))
    }

    # -------------------
    # Similarity Functions
    # -------------------
    def embedding_similarity(dist):
        return np.exp(-dist)

    def jaccard_similarity(set_a, set_b):
        inter = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return inter / union if union != 0 else 0.0

    def category_similarity(cat_i, cat_j):
        return jaccard_similarity(cat_i, cat_j)

    # Weights for Final Similarity
    w_feature = 0.4
    w_rating  = 0.2
    w_rated   = 0.1
    w_man     = 0.1
    w_cat     = 0.2

    threshold = 0.5
    num_neighbours = 10

    # -------------------
    # GPU-based distance function (batch-wise)
    # -------------------
    def compute_gpu_distances(batch_embs: torch.Tensor, all_embs: torch.Tensor) -> torch.Tensor:
        """
        batch_embs: shape (batch_size, dim)
        all_embs: shape (num_items, dim)
        returns: L2 distances shape (batch_size, num_items)
        """
        with torch.no_grad():
            dists = torch.cdist(batch_embs, all_embs, p=2)
        return dists

    # -------------------
    # Parallel batch processing
    # -------------------
    def process_batch(batch_indices):
        batch_results = []

        batch_embs = item_embs_tensor[batch_indices]

        distances = compute_gpu_distances(batch_embs, item_embs_tensor)

        distances_cpu = distances.cpu()

        for idx, i in enumerate(batch_indices):
            F_i = item_features_list[i]
            rating_i = avg_ratings[i]
            rated_i = rated_ids[i]
            man_i = man_ids[i]
            cat_i = category_map[i]

            candidate_mask = np.abs(avg_ratings - rating_i) <= 0.5
            candidate_indices = np.where(candidate_mask)[0]

            if len(candidate_indices) == 0:
                continue

            similarities = []
            for j in candidate_indices:
                if j == i:
                    continue

                F_j = item_features_list[j]
                feature_jacc = jaccard_similarity(F_i, F_j)
                rating_j = avg_ratings[j]
                rating_sim = 1.0 - abs(rating_i - rating_j) / 5.0
                rated_j = rated_ids[j]
                rated_sim = 1.0 if (rated_i == rated_j and rated_i != -1) else 0.0
                man_j = man_ids[j]
                man_sim = 1.0 if (man_i == man_j and man_i != -1) else 0.0
                cat_j = category_map[j]
                cat_sim = category_similarity(cat_i, cat_j)

                metadata_sim = (w_feature * feature_jacc +
                                w_rating  * rating_sim  +
                                w_rated   * rated_sim   +
                                w_man     * man_sim     +
                                w_cat     * cat_sim)

                if metadata_sim >= threshold:
                    similarities.append((j, metadata_sim))

            if not similarities:
                continue

            similarities.sort(key=lambda x: x[1], reverse=True)
            top_50 = [idx for idx, _ in similarities[:50]]

            if not top_50:
                continue

            dists = distances_cpu[idx][top_50].numpy()  # (top_50,)

            emb_sims = embedding_similarity(dists)  # (top_50,)

            final_sims = np.array([alpha * sim + beta * emb_sim for (_, sim), emb_sim in zip(similarities[:50], emb_sims)])

            top_10_idx = final_sims.argsort()[::-1][:num_neighbours]
            top_10 = [(i, top_50[j], final_sims[j]) for j in top_10_idx]

            batch_results.extend(top_10)

        return batch_results

    # -------------------
    # Divide item indices into batches of size 64 and process in parallel
    # -------------------
    batch_size = 512
    top_10_neighbors = []

    all_indices = list(range(len(metadata)))  # Process all items
    batches = [all_indices[i:i + batch_size] for i in range(0, len(all_indices), batch_size)]

    results = []
    max_workers = min(14, os.cpu_count() or 1)
    print(f"Processing items in parallel with {max_workers} workers, batch_size={batch_size}...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_batch, batch): batch for batch in batches}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Batches"):
            batch_result = future.result()
            top_10_neighbors.extend(batch_result)

    # -------------------
    # Build item-item 'similar' edges in the HeteroData graph
    # -------------------
    if len(top_10_neighbors) == 0:
        data['item', 'similar', 'item'].edge_index = torch.empty((2,0), dtype=torch.long)
        data['item', 'similar', 'item'].edge_attr  = torch.empty((0,1),  dtype=torch.float)
    else:
        item_i, item_j, edge_attr_list = zip(*top_10_neighbors)
        data['item', 'similar', 'item'].edge_index = torch.tensor([item_i, item_j], dtype=torch.long)
        data['item', 'similar', 'item'].edge_attr  = torch.tensor(edge_attr_list, dtype=torch.float).unsqueeze(1)

    print("Graph construction complete!")
    print(data)

    # -------------------
    # Save the final graph with dynamic filename based on alpha and beta
    # -------------------
    filename = f'filtered_results/hetero_graph_small_{str(alpha).replace(".", "")}_{str(beta).replace(".", "")}.pt'
    torch.save(data, filename)
    print(f"Graph saved as '{filename}'.")