
import os
import pickle
import random
import argparse
from typing import List, Tuple

from tqdm import tqdm
import numpy as np


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def create_genuine_pairs( 
        clustering_location: str = "",
        embedding_location: str = "",
    ):
    """Creates a list of genuine pairs given clustering and embedding data

    Args:
        clustering_location (str, optional): Location of clustering data.
        embedding_location (str, optional): Location of embedding location.

    Returns:
        list: A list of genuine pairs presented as tuples (sample1: str, sample2: str, sample_similarity: float)
    """

    with open(embedding_location, "rb") as pkl_in:
        embeddings = pickle.load(pkl_in)

    with open(clustering_location, "rb") as pkl_in:
        cluster_data = pickle.load(pkl_in)

    genuine_pairs = list()
    for (identitiy, clusters) in tqdm(cluster_data.items()):
        for cluster_id, cluster_samples in clusters.items():
            for sample in cluster_samples:
                sample_pairs = [(sample, (gp := random.choice(cluster_samples_)), cosine_similarity(embeddings[sample], embeddings[gp])) for cluster_id_, cluster_samples_ in clusters.items() if cluster_id_ != cluster_id]
                genuine_pairs.extend(sample_pairs)
    
    return list(set(genuine_pairs))


def calculate_indexes(
        base_quality_location: str = "",
        clustering_location: str = "",
        embedding_location: str = ""
    ):
    """Calculates indices for quality optimization

    Args:
        base_quality_location (str, optional): Location of base quality scores. 
        clustering_location (str, optional): Location of clustering data.
        embedding_location (str, optional): Location of embedding location. 

    Returns:
        tuple: (quality_indices: dict, sample_gp_indices: dict, quality_data: list)
    """

    genuine_pairs = create_genuine_pairs(clustering_location=clustering_location,
                                         embedding_location=embedding_location)

    with open(base_quality_location, "rb") as pkl_in:
        quality_data = pickle.load(pkl_in)
    quality_data = list(quality_data.items())
    quality_data.sort(key=lambda x: x[1])

    quality_indices = {sample: i for i, (sample, _) in enumerate(quality_data)}

    genuine_pairs.sort(key=lambda x: x[2])
    sample_gp_indices = {}
    for i, (sample1, sample2, _) in enumerate(tqdm(genuine_pairs)):
        if quality_indices[sample1] < quality_indices[sample2]:
            sample_gp_indices.setdefault(sample1, []).append(i)
        else:
            sample_gp_indices.setdefault(sample2, []).append(i)
    
    sample_gp_indices = {k: sum(v)/len(v) for k, v in sample_gp_indices.items()}

    return (quality_indices, sample_gp_indices, quality_data)


def optimize(
        epsilon: float = 0.001,
        iterations: int = 10,
        base_quality_location: str = "",
        clustering_location: str = "",
        embedding_location: str = ""
    ):
    """
    Main optimization function generating improved quality scores.

    Args:
        epsilon (float, optional): Epsilon determining how much scores are adjusted. Defaults to 0.001.
        iterations (int, optional): How many iterations of the optimization to average over. Defaults to 10.
        base_quality_location (str, optional): Location of base quality scores. 
        clustering_location (str, optional): Location of clustering data.
        embedding_location (str, optional): Location of embedding location. 

    Returns:
        dict: Improved quality scores.
    """

    final_quality = {}
    quality_distribution = None
    for iteration in range(iterations):
        print(f" Runnning {iteration}/{iterations}", flush=True)

        quality_indices, sample_gp_indices, quality_distribution = calculate_indexes(base_quality_location=base_quality_location,
                                                                                     clustering_location=clustering_location,
                                                                                     embedding_location=embedding_location)

        for (sample, q_index) in tqdm(quality_indices.items()):
            final_quality.setdefault(sample, []).append(q_index + epsilon * (sample_gp_indices[sample] - q_index) if sample in sample_gp_indices else q_index)

    final_quality = [(k, sum(v)/len(v)) for k, v in final_quality.items()]
    final_quality.sort(key=lambda x: x[1])

    return {sample: quality for ((sample, _), (_, quality)) in zip(final_quality, quality_distribution)}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-ql", "--quality_location", type=str, required=True, help="Location of quality scores of dataset, to be improved.")
    parser.add_argument("-sl", "--save_location", type=str, required=True, help="Location to store improved scores of dataset in.")
    parser.add_argument("-cl", "--clustering_location", type=str, required=True, help="Location of the identitiy clustering data of the dataset.")
    parser.add_argument("-el", "--embedding_location", type=str, required=True, help="Location of the image embeddings of the dataset.")
    parser.add_argument("-ep", "--epsilon", type=float, default=0.001)
    parser.add_argument("-it", "--iterations", type=int, default=10)
    args = parser.parse_args()

    assert os.path.exists(args.quality_location), f"Quality scores path '{args.quality_location}' does not exist!"
    assert os.path.exists(args.clustering_location), f"Clustering data path '{args.clustering_location}' does not exist!"
    assert os.path.exists(args.embedding_location), f"Image embedding path '{args.embedding_location}' does not exist!"
    assert 1. > args.epsilon > 0., f"Epsilon should be in range (0., 1.)!"
    assert args.iterations > 0, f"Iterations should be positive!"

    quality_scores = optimize(epsilon=args.epsilon, 
                              iterations=args.iterations, 
                              base_quality_location=args.quality_location,
                              clustering_location=args.clustering_location,
                              embedding_location=args.embedding_location)

    with open(args.save_location, "wb") as pkl_out:
        pickle.dump(quality_scores, pkl_out)
