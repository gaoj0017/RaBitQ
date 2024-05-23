
import numpy as np
import faiss
import struct
import os
from utils.io import *

source = './'

if __name__ == '__main__':

    dataset = 'sift'
    print(f"Clustering - {dataset}")
    # path
    path = os.path.join(source, dataset)
    data_path = os.path.join(path, f'{dataset}_base.fvecs')
    X = read_fvecs(data_path)
    D = X.shape[1]    
    K = 4096
    centroids_path = os.path.join(path, f'{dataset}_centroid_{K}.fvecs')
    dist_to_centroid_path = os.path.join(path, f'{dataset}_dist_to_centroid_{K}.fvecs')
    cluster_id_path = os.path.join(path, f'{dataset}_cluster_id_{K}.ivecs')

    # cluster data vectors
    index = faiss.index_factory(D, f"IVF{K},Flat")
    index.verbose = True
    index.train(X)
    centroids = index.quantizer.reconstruct_n(0, index.nlist)
    dist_to_centroid, cluster_id = index.quantizer.search(X, 1)
    dist_to_centroid = dist_to_centroid ** 0.5

    to_fvecs(dist_to_centroid_path, dist_to_centroid)
    to_ivecs(cluster_id_path, cluster_id)
    to_fvecs(centroids_path, centroids)
