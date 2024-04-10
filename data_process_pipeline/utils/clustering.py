import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


def hac_clustering(real_items, embeddings, threshold):
    if len(real_items) == 0:
        return []
    if len(real_items) == 1:
        return [real_items], None

    # Normalize the embeddings to unit length
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Perform clustering
    clustering_model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="cosine",
        linkage="complete",
    )
    clustering_model.fit(embeddings)
    cluster_assignment = clustering_model.labels_

    # calculate score
    try:
        score = float(silhouette_score(embeddings, cluster_assignment))
    except:
        score = None

    id2cluster = {}
    for item_idx, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in id2cluster:
            id2cluster[cluster_id] = []
        id2cluster[cluster_id].append(real_items[item_idx])

    return sorted(id2cluster.values(), key=lambda c: len(c), reverse=True), score


# def hac_clustering_retain_index(real_items, embeddings, threshold):
#     if len(real_items) == 0:
#         return []
#     if len(real_items) == 1:
#         return [real_items]

#     # Normalize the embeddings to unit length
#     embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

#     # Perform clustering
#     clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, metric="cosine", linkage="complete")
#     clustering_model.fit(embeddings)
#     cluster_assignment = clustering_model.labels_

#     id2cluster = {}
#     for item_idx, cluster_id in enumerate(cluster_assignment):
#         if cluster_id not in id2cluster:
#             id2cluster[cluster_id] = []
#         id2cluster[cluster_id].append((item_idx, real_items[item_idx]))

#     return sorted(id2cluster.values(), key=lambda c: len(c), reverse=True)


def hac_clustering_retain_index(real_items, embeddings, threshold):
    if len(real_items) == 0:
        return [], None
    if len(real_items) == 1:
        return [real_items], None

    print(f"all embeddings: {len(embeddings)}")
    # Normalize the embeddings to unit length
    embeddings_normalized = embeddings / np.linalg.norm(
        embeddings, axis=1, keepdims=True
    )

    # Identify unique embeddings and their indices
    unique_embeddings = np.unique(embeddings_normalized, axis=0)

    print(f"unique embeddings: {len(unique_embeddings)}")

    # Perform clustering on unique embeddings
    clustering_model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="cosine",
        linkage="complete",
    )
    clustering_model.fit(unique_embeddings)
    unique_cluster_assignment = clustering_model.labels_

    # calculate score
    try:
        score = float(silhouette_score(unique_embeddings, unique_cluster_assignment))
    except:
        score = None
    # Map unique clustering results back to original indices
    id2cluster = {}
    for unique_idx, cluster_id in enumerate(unique_cluster_assignment):
        if cluster_id not in id2cluster:
            id2cluster[cluster_id] = []
        # Find all original indices that match the unique index
        original_indices = np.where(
            (embeddings_normalized == unique_embeddings[unique_idx]).all(axis=1)
        )[0]
        for original_idx in original_indices:
            id2cluster[cluster_id].append((original_idx, real_items[original_idx]))

    return sorted(id2cluster.values(), key=lambda c: len(c), reverse=True), score


def secondary_clustering(items_with_indices, embeddings, threshold):
    # 'items_with_indices' is a list of tuples: [(original_index, item), ...]

    if len(items_with_indices) <= 1:
        return [items_with_indices], None

    # Normalize the embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Perform clustering
    clustering_model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="cosine",
        linkage="average",
    )
    clustering_model.fit(embeddings)
    cluster_assignment = clustering_model.labels_

    # calculate score
    try:
        score = float(silhouette_score(embeddings, cluster_assignment))
    except:
        score = None

    id2cluster = {}
    for idx, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in id2cluster:
            id2cluster[cluster_id] = []
        id2cluster[cluster_id].append(items_with_indices[idx])

    return sorted(id2cluster.values(), key=lambda c: len(c), reverse=True), score
