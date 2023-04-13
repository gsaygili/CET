"""
This Python module defines several distance measures functions to be used in confidence estimation algorithm. 

Available distance measures:
  - Euclidean distance
  - Manhattan distance
  - Chebyshev distance
  - Lorentzian distance
  - Canberra distance
  - Bray Curtis distance
  - Cosine distance
  - Chord distance
  - Jaccard distance
  - Dice distance
  - Squared Chord distance
  - Vicis symmetric distance
  - Divergence distance
  - Clark distance

Usage:
The distance measures can be called by passing a 2D NumPy array to the respective function.
The functions return the distance matrix as a 1D NumPy array.

"""

import scipy.spatial.distance as dist
import numpy as np
from numpy import abs
from itertools import combinations



#-----------MINKOVSKI DISTANCE MEASURES---------------
#1-EUCLIDEAN
def euclidean(A):
    dist2 = dist.pdist(A, "euclidean")
    return dist2

#2-MANHATTAN
def manhattan(A):
    dist2 = dist.pdist(A, "cityblock")
    return dist2

#3-CHEBYSHEV
def chebyshev(A):
    dist2 = dist.pdist(A, "chebyshev")
    return dist2

#-------------L1 DISTANCE MEASURES---------------
#4-LORENTZIAN
def lorentzian(A):
    dist2 = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        x = A[index[0],:]
        y = A[index[1],:]
        dist2[j] = np.sum(np.log(1+np.abs(x-y)))
        j+=1 
    return dist2

#5-CANBERRA
def canberra(A):
    dist2 = dist.pdist(A, "canberra")
    return dist2

#6-SORENSEN(BRAY_CURTIS)
def braycurtis(A):
    dist2 = dist.pdist(A, "braycurtis")
    return dist2

#-------------INNER PRODUCT DISTANCE MEASURES--------------
#7 COSINE
def cosine(A):
    dist2 = dist.pdist(A, "cosine")
    return dist2

#8-CHORD
def chord(A):
    dist2 = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        x = A[index[0],:]
        y = A[index[1],:]
        top = (x*y).sum()
        bot = np.sqrt((x**2).sum()) * np.sqrt((y**2).sum())
        dist2[j] = np.sqrt(2-2*(top/bot))
        j+=1        
    return dist2

#9-JACCARD
# jaccard distance is calculated between two boolean 1-D arrays. 

def jaccard(A):   
    dist2 = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        x = A[index[0],:]
        y = A[index[1],:]
        top = ((x-y)**2).sum()
        bot = (x**2).sum() + (y**2).sum() - (x*y).sum()
        dist2[j] = top/bot
        j+=1        
    return dist2

#10-DICE
# Dice distance is calculated between two boolean 1-D arrays. 
def dice(A):   
    dist2 = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        x = A[index[0],:]
        y = A[index[1],:]
        top = 2*((x*y).sum())
        bot = (x**2).sum() + (y**2).sum()
        dist2[j] = 1 - top/bot
        j+=1        
    return dist2

#-------------SQUARED CHORD DISTANCE MEASURES--------------
#11-Squared Chord
def squared_chord(A, axis=0, keepdims=False):
    dist2 = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        p = A[index[0],:]
        q = A[index[1],:]
        p=np.maximum(p, 1e-12)
        q=np.maximum(q, 1e-12)
        dist2[j] = ((np.sqrt(p)-np.sqrt(q))**2).sum()
        j+=1
    return dist2

#-------------Vicissitude Distance Measures -------------
#12-Vicis symmetric 
def vicis_symmetric(A):
    dist1 = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        x = A[index[0],:]
        y = A[index[1],:]
        top = (x-y)**2
        bot = (np.min(x-y))**2
        dist1[j] = np.sum(top/bot)
        j+=1        
    return dist1

#13-DIVERGENCE
def divergence(A):
    dist2 = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        x = A[index[0],:]
        y = A[index[1],:]        
        dist2[j] = 2*((((x-y)**2)/((x+y)**2+1e-12)).sum())
        j+=1        
    return dist2

#14-CLARK
def clark(A):
    dist2 = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        x = A[index[0],:]
        y = A[index[1],:]
        dist2[j] = np.sqrt(np.sum(((x-y)/(np.abs(x)+np.abs(y)+1e-12))**2))
        j+=1        
    return dist2

#15-Squared Euclidean 
def squared_euclidean(A):
    dist2 = dist.pdist(A, "sqeuclidean")    
    return dist2

#16-Averaged Euclidean

def average_euclidean(A):
    dist2 = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        x = A[index[0],:]
        y = A[index[1],:]
        dist2[j] = np.sqrt((np.sum((x-y)**2))/len(x)) #bana len(x)'e bolmek daha mantikli geliyor.
        j+=1        
    return dist2

#17-Squared Chi-Squared
def chi_squared(A):
    dist2 = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        x = A[index[0],:]
        y = A[index[1],:]
        top = (x-y)**2
        bot = np.abs(x+y)+1e-12
        dist2[j] = np.sum(top/bot)
        j+=1        
    return dist2

#18-Mean Censored Euclidean
def mean_cen_euc(A):
    dist2 = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        x = A[index[0],:]
        y = A[index[1],:]
        top = np.sum((x-y)**2)
        bot = len([s for s in x**2+y**2 if s != 0])
        dist2[j] = np.sqrt(top/bot)
        j+=1        
    return dist2

#19-JENSEN DIFFERENCE
from scipy.stats import entropy
from scipy import special
def jensendifference(A):
    dist1 = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        p = A[index[0],:]
        q = A[index[1],:]
        p=np.maximum(p, 1e-12)
        q=np.maximum(q, 1e-12)
        m = (p + q) / 2.0
        left = (special.entr(p)+special.entr(q))/2
        right = special.entr(m)
        jd = np.sum(left-right, axis=0)
        dist1[j] = np.abs(jd / 2.0)
        j+=1
    return dist1

from scipy.special import rel_entr
#20-JEFFREYS
def jeffreys(A):
    dist2 = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        p = A[index[0],:]
        q = A[index[1],:]
        p=np.maximum(p, 1e-12)
        q=np.maximum(q, 1e-12)
        left = p-q
        right = rel_entr(1,(q/p))
        dist2[j] = np.sum(left*right, axis=0)
        j+=1
    return dist2

#21-KullbackLeibler
def kullbackleibler(A):
    dist1 = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        p = A[index[0],:]
        q = A[index[1],:]
        p=np.maximum(p, 1e-12)
        q=np.maximum(q, 1e-12)
        kl = entropy(p,q)
        dist1[j] = kl
        j+=1        
    return dist1

#------------------OTHER DISTANCE MEASURES--------------
#22-Average Distance
def average_dist(A):
    manh = dist.pdist(A, "cityblock") 
    cheb = dist.pdist(A, "chebyshev")
    dist2 = (manh + cheb)/2
    return dist2

#23-Whittakerâ€™s index of association
def WIAD(A):
    dist2 = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        x = A[index[0],:]
        y = A[index[1],:]
        left = x/(x.sum())
        right = y/(y.sum())
        dist2[j] = np.sum(np.abs(left-right))/2
        j+=1        
    return dist2

#24-Squared Pearson Distance
def squared_pearson(A):
    dist2 = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        x = A[index[0],:]
        y = A[index[1],:]
        corr, _ = pearsonr(x, y)
        dist2[j] = 1-(corr**2)
        j+=1        
    return dist2

#25-CORRELATION
#Compute the correlation distance between two 1-D arrays.
def correlation(A):
    dist2 = dist.pdist(A, "correlation") #formulde 2ye boluyor ama bunlar bolmemis
    return dist2

#26-Pearson Distance
from scipy.stats import pearsonr
def pearson(A):
    dist2 = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        x = A[index[0],:]
        y = A[index[1],:]
        corr, _ = pearsonr(x, y)
        dist2[j] = 1-(corr)
        j+=1        
    return dist2

#27-Motyka Distance
def motyka(A):
    dist2 = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        x = A[index[0],:]
        y = A[index[1],:]
        top = (np.maximum(x, y)).sum() 
        bot = (x+y).sum()
        dist2[j] = np.abs(top/bot)
        j+=1        
    return dist2

#28-HASSANAT
def hassanat(k):   
    x1 = []
    x1.extend([[p1,p2] for p1,p2 in combinations (k[:,0] ,2)])
    y1 = []
    y1.extend([[p1,p2] for p1,p2 in combinations (k[:,1] ,2)])
    dist = np.zeros(len(x1))
    for i in range(len(x1)):
        x = [x1[i]]
        y = [y1[i]]
        total = 0
        for xi, yi in zip(x, y):
            min_value = np.min([xi, yi])
            max_value = np.max([xi, yi])
            total += 1  # we sum the 1 in both cases
            if min_value >= 0:
                total -= (1 + min_value) / (1 + max_value)
            else:
                # min_value + abs(min_value) = 0, so we ignore that
                total -= 1 / (1 + max_value + abs(min_value))                
        dist[i] = total
    return dist

#29-Mutual Information Distance 
from sklearn.metrics import mutual_info_score
def mutual_information(A):
    dist2 = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        x = A[index[0],:]
        y = A[index[1],:]
        sim = mutual_info_score(x, y)
        dist2[j] = 1-sim
        j+=1        
    return dist2


# _all_= [
#     'euclidean',
#     'manhattan',
#     'chebyshev',
#     'lorentzian',
#     'canberra',
#     'braycurtis',
#     'cosine',
#     'chord',
#     'jaccard',
#     'dice',
#     'squared_chord',
#     'vicis_symmetric',
#     'divergence',
#     'clark',
#     'squared_euclidean',
#     'average_euclidean',
#     'chi_squared',
#     'mean_cen_euc',
#     'jensendifference',
#     'jeffreys',
#     'kullbackleibler',
#     'average_dist',
#     'WIAD',
#     'squared_pearson'
#     'correlation',
#     'pearson',
#     'motyka',
#     'hassanat',
#     'mutual_information']
