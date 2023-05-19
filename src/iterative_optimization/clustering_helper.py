
import os
import pickle
import argparse

import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

class VGGFace2Clustering():

    def __init__(
        self,
        location: str = "",
        embedding_location: str = "",
        number_of_clusters: int = 20,
        save_location : str = ""
    ):
        super().__init__()

        self.location = location
        self.embedding_location = embedding_location
        self.number_of_clusters = number_of_clusters
        self.save_location = save_location

    
    def prepare_id_clusters(self):

        with open(self.embedding_location, "rb") as pkl_in:
            embeddings = pickle.load(pkl_in)
        
        embedding_keys, embeddings = zip(*embeddings.items())
        embeddings = normalize(np.array(embeddings))
        embedding_keys = {key: i for i, key in enumerate(embedding_keys)}
        reverse_embeddings_keys = {i: key for key, i in embedding_keys.items()}

        print(f"=" * 100 + "\n\n BUILDING CLUSTER LIST FOR ALL IDENTITIES \n\n" + f"=" * 100)
        items_per_id = {}
        for (dir, subdirs, files) in tqdm(os.walk(self.location)):
            if files:
                id = dir.split("/")[-1]
                indexes = list(map(lambda x: embedding_keys[os.path.join(dir, x).split("arcface/")[1]], files)) # Change .split parameter here so that keys match keys in embedding file
                id_embeddings = embeddings[indexes]
                id_kmeans = KMeans(n_clusters=self.number_of_clusters, random_state=0).fit(id_embeddings)
                per_cluster_ids = {}
                for _id, cluster in list(zip(indexes, id_kmeans.labels_)):
                    per_cluster_ids.setdefault(cluster, []).append(_id)
                items_per_id[id] = per_cluster_ids

        for id, clusters in items_per_id.items():
            for cluster_id, cluster_items in clusters.items():
                items_per_id[id][cluster_id] = [reverse_embeddings_keys[item] for item in cluster_items]
        
        with open(self.save_location, "wb") as pkl_out:
            pickle.dump(items_per_id, pkl_out)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-il", "--image_location", type=str, required=True, help="Location of VGGFace2 train and test folders.")
    parser.add_argument("-el", "--embedding_location", type=str, required=True, help="Location of extracted embeddings for the VGGFace2 dataset.")
    parser.add_argument("-cpi", "--cluster_per_individual", type=int, default=20)
    parser.add_argument("-sl", "--save_location", type=str, required=True, help="Location where to store clustering data.")
    args = parser.parse_args()

    assert os.path.exists(args.image_location), f" Image location {args.image_location} does not exist!"
    assert os.path.exists(args.embedding_location), f" Embedding location {args.embedding_location} does not exist!"
    assert args.cluster_per_individual > 1, f" Number of cluster per person must be larger than 1!"

    clustering = VGGFace2Clustering(location=args.image_location, 
                                    embedding_location=args.embedding_location,
                                    number_of_clusters=args.cluster_per_individual,
                                    save_location=args.save_location)
    clustering.prepare_id_clusters()

