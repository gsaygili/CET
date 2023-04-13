

from math import*
import scipy.spatial.distance as dist
import numpy as np
from numpy import (array, zeros, logical_and, logical_or, logical_xor, where,
                   mean, std, argsort, take, ravel, logical_not, shape, sqrt, abs,
                   sum, square, asmatrix, asarray, multiply, min, any, all, isfinite,
                   nonzero, nan_to_num, geterr, seterr, isnan)
from itertools import combinations
import time
from distance_measures import cosine, braycurtis, dice, correlation, pearson, WIAD, kullbackleibler


def extract_feats(emb_folder, distance_measures, K=20):  
        
    X_emb = np.load(emb_folder + "X_emb.npy",allow_pickle=True)
    X = np.load(emb_folder + "X.npy",allow_pickle=True)
    features = np.zeros((X.shape[0], K, len(distance_measures)))
    X_d = dist.squareform(dist.pdist(X_emb, "euclidean"))
    sort_index_d = np.argsort(X_d)
    file1 = open(emb_folder + "time_file.txt","w")
    file1.close()
    for ind, distname in enumerate(distance_measures):
       file1 = open(emb_folder + "time_file.txt","a")
       file1.writelines("Calculating " + distname + " Distances \n")
       print("Calculating " + distname + " Distances")
       start_time = time.time()        
       X_D = dist.squareform(eval(distname + "(X)")) 
       X_D = np.nan_to_num(X_D)
       X_D = (X_D - np.min(X_D))/np.ptp(X_D)
       sort_D = np.sort(X_D)
       for i in range(X_D.shape[0]):
            cost = X_D[i, :]
            s_D = sort_D[i, 1:K + 1]
            s_d = np.sort(cost[sort_index_d[i, 1:K + 1]])
            features[i, :, ind] = np.abs(s_D - s_d)

       print(np.any(np.isnan(features)))
       file1.writelines("--- takes: " + str(time.time() - start_time) + " seconds ---\n")
       print("--- takes: %s seconds ---" % (time.time() - start_time))
       file1.close()
    np.save(emb_folder + 'X_feat.npy', features)
	 
'''
_all_= ['cosine', 'braycurtis', 'dice', 'correlation', 'pearson','WIAD', 'kullbackleibler']
emb_folder = "D:/BusraOYigin/Research/t-SNE/Datasets/Baron_Human/UMAP/"

extract_feats(emb_folder)
'''
