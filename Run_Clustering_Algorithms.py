

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
import scanpy as sc # import scanpy to handle our AnnData 
from sklearn.cluster import DBSCAN
from sklearn.metrics import confusion_matrix
from munkres import Munkres
from sklearn.neighbors import NearestNeighbors

#%% Loading dataset for AMB18 or Baron Human training results

def load_data(dataset_name, emb_folder):
    filename = emb_folder + dataset_name + '/'
    X = np.load(filename + "X.npy", allow_pickle=True) #original dataset    
    y = np.load(filename + "y_pred.npy", allow_pickle=True) #predicted conf scores
    y_true = np.load(filename + "y_conf_le.npy", allow_pickle=True) #ground truth labels after label encoding

    adata = sc.AnnData(X, dtype=X.dtype)

    # Normalize counts per cell. Normalize each cell by total counts over all genes, so that every cell has the same total count after normalization.
    sc.pp.normalize_total(adata)
    # Logarithmize the data matrix.
    sc.pp.log1p(adata)

    return X, y, y_true

#%%

def calculate_thr (y, thr_ratio = 0.6):
    num_sample = int(np.floor(len(y)*thr_ratio)) #it calculates the 60% (default) of the total samples
    sorted_y = np.sort(y)[::-1] #sort in ascending order
    thr = sorted_y[num_sample] # get the value of 'num_sample' as thr
    return thr

def plot_clusters(X, y_kmeans, y_true, save_file, fig_no):
    plt.figure()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    if (fig_no==0):
        ax1.set_title("without any elimination")
        ax2.set_title("without any elimination with true targets")
        name = 'we'
    if (fig_no==1):
        ax1.set_title("only low confident samples")
        ax2.set_title("only low confident samples with true targets")
        name = 'oc'
    if (fig_no==2):
        ax1.set_title("after eliminating low confidence samples")
        ax2.set_title("after eliminating low confidence samples with true targets")
        name = 'ae'
    fig.set_size_inches(18, 7)  
    ax1.scatter(X[:, 0], X[:, 1], c = y_kmeans, s=50, cmap='viridis')
    ax2.scatter(X[:, 0], X[:, 1], c = y_true, s=50, cmap='viridis')
    plt.savefig(save_file+'_DBSCAN_clusters_'+ name+'.pdf')
    
#%% Accuracy calculation for evaluation of the clustering model

# inspired by https://smorbieu.gitlab.io/accuracy-from-classification-to-clustering-evaluation/#:~:text=Accuracy%20is%20often%20used%20to,is%20also%20used%20for%20clustering.
def _make_cost_m(cm):
    s = np.max(cm)
    return (- cm + s)

#inspired by Hungarian (Munkres) algorithm: https://software.clapper.org/munkres/
def cal_accuracy(cm):
    m = Munkres()
    indexes = m.compute(_make_cost_m(cm)) #it applies Hungarian algorithm
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]
    acc = np.trace(cm2) / np.sum(cm2)
    return acc


#%% Functions to choose hyperparameters for DBSCAN and K-Means algorithm
from kneed import KneeLocator

def best_elbow(X):
    Sum_of_squared_distances = []
    K = range(3,16)
    for num_clusters in K :
        kmeans = KMeans(n_clusters = num_clusters)
        kmeans.fit(X)
        Sum_of_squared_distances.append(kmeans.inertia_)        
    plt.figure() 
    x = range(1, len(Sum_of_squared_distances)+1)
    kn = KneeLocator(x, Sum_of_squared_distances, curve='convex', direction='decreasing')
    plt.xlabel('number of clusters k')
    plt.ylabel('Sum of squared distances')
    plt.plot(x, Sum_of_squared_distances, 'bx-')
    plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', color='r')
    plt.title('Elbow Method For Optimal k')
    plt.show()     
    return kn.knee 

def dbscan_predict(model, X):
    nr_samples = X.shape[0]
    y_new = np.ones(shape=nr_samples, dtype=int) * -1
    for i in range(nr_samples):
        diff = model.components_ - X[i, :]  
        dist = np.linalg.norm(diff, axis=1)  
        shortest_dist_idx = np.argmin(dist)
        if dist[shortest_dist_idx] < model.eps:
            y_new[i] = model.labels_[model.core_sample_indices_[shortest_dist_idx]]
    return y_new


# A low minPts means it will build more clusters from noise, so don't choose it too small.
# Generally, MinPts should be greater than or equal to the dimensionality of the data set. For 2-dimensional data, use DBSCAN’s default value of MinPts = 4 (Ester et al., 1996).
# If your data has more than 2 dimensions, choose MinPts = 2*dim, where dim= the dimensions of your data set (Sander et al., 1998).
def find_best_MinPts_eps(X):
    #decide MinPts
    MinPts = len(X.shape)*2 + round(X.shape[0]/1000) #the larger the dataset, the larger the value    
    neighbors = NearestNeighbors(n_neighbors=20)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)
    distances_sorted = np.sort(distances, axis=0)
    sorted_distance = distances_sorted[:,-1]
    x = range(1, len(sorted_distance)+1)
    kn = KneeLocator(x, sorted_distance, curve='convex', direction='increasing')
    eps=sorted_distance[kn.knee]
    
    return MinPts, eps
    
#%% Run DBSCAN Clustering Experiments

Datasets = ['Baron_Human', 'AMB18','Segerstolpe', 'Baron_Mouse', 'CellBench', 'Muraro'] # 
num_classes = [14, 15, 13, 13, 5, 9] #
repeat_number = 1 #no need to repeat for DBSCAN
col_names = ['dataset','num_class','chosen_thr', 'conf_thr', 'ARI_we','ARI_oc','ARI_ae','NMI_we','NMI_oc','NMI_ae','ACC_we','ACC_oc','ACC_ae', 'num_clas_we', 'num_clas_oc', 'num_class_ae']

all_results_list = []

elimination_rate=0.9

if elimination_rate == 0.8:
    elm_sym = '0_8'
if elimination_rate == 0.7:
    elm_sym = '0_7' 
if elimination_rate == 0.9:
    elm_sym = '0_9'
if elimination_rate == 0.6:
    elm_sym = '0_6'
    
#for loop to obtain the results for each dataset    
for dataset, num_class in zip(Datasets, num_classes):
    all_ARI_we = [] #ARI score without elimination (with whole samples)
    all_ARI_oc = [] #ARI scores with only confident values
    all_ARI_ae = [] #ARI scores after elimination
    all_NMI_we = [] #NMI score without elimination (with whole samples)
    all_NMI_oc = [] #NMI scores with only confident values
    all_NMI_ae = [] #NMI scores after elimination
    all_ACC_we = [] #ACC score without elimination (with whole samples)
    all_ACC_oc = [] #ACC scores with only confident values
    all_ACC_ae = [] #ACC scores after elimination
    
    #load dataset
    X, y, y_true = load_data(dataset)
    
    #Calculate the corresponding threshold value for the confidence score of the elimination ratio
    thr = calculate_thr(y, elimination_rate)
    print('thr:',thr)
    
    #number of components for PCA
    n_c = 30
    #apply PCA dim-red before clustering
    pca = PCA(n_components = n_c)
    
    X_embedded = pca.fit_transform(X)
    # Low-confident samples will be eliminated according to the thr value obtained.
    X_conf_emb = X_embedded[y>thr] #eliminated version of the data
    #print(X_embedded.shape[1])
    
    for j in range(repeat_number):
        
        ###############################
        # Without Confidence Scores - Without any elimination 
        ###############################
        minPts, best_eps = find_best_MinPts_eps(X_embedded) 
        print('eps:', best_eps)
        print('minPts:', minPts)
        dbscan = DBSCAN(min_samples = minPts, eps=best_eps)        
        clusters = dbscan.fit_predict(X_embedded)
        #calculate the evaluation metrics
        ARI_we = adjusted_rand_score(y_true, clusters)
        NMI_we = normalized_mutual_info_score(y_true, clusters)
        cm = confusion_matrix(y_true, clusters)
        ACC_we = cal_accuracy(cm)
        print(ARI_we,NMI_we,ACC_we)
        #append the scores 
        all_ACC_we.append(ACC_we)
        all_ARI_we.append(ARI_we)
        all_NMI_we.append(NMI_we)
        
        ###############################
        # With only Confident Samples
        ###############################
        minPts2, best_eps2 = find_best_MinPts_eps(X_conf_emb) 
        print('eps:', best_eps2)
        print('minPts:', minPts2) 
        dbscan2 = DBSCAN(min_samples = minPts2, eps=best_eps2)
        clusters2 = dbscan2.fit_predict(X_conf_emb)
        
        #calculate the evaluation metrics
        ARI_oc = adjusted_rand_score(y_true[y>thr], clusters2)
        NMI_oc = normalized_mutual_info_score(y_true[y>thr], clusters2)
        cm2 = confusion_matrix(y_true[y>thr], clusters2)
        ACC_oc = cal_accuracy(cm2)
        print(ARI_oc,NMI_oc,ACC_oc)
        #append the scores 
        all_ACC_oc.append(ACC_oc)
        all_ARI_oc.append(ARI_oc)
        all_NMI_oc.append(NMI_oc)
        
        ###############################
        # With Confidence Scores - After Elimination using the trained model on not eliminated data
        ###############################
        # Predict the clusters with the trained model
        P = dbscan_predict(dbscan2, X_embedded)
        #calculate the evaluation metrics
        ARI_ae = adjusted_rand_score(y_true, P)
        NMI_ae = normalized_mutual_info_score(y_true, P)
        cm3 = confusion_matrix(y_true, P)
        ACC_ae = cal_accuracy(cm3)
        print(ARI_ae,NMI_ae,ACC_ae)
        #append the scores
        all_ACC_ae.append(ACC_ae)
        all_ARI_ae.append(ARI_ae)
        all_NMI_ae.append(NMI_ae)
        
    #visualizing the clustering results
    #plot clusters
    we_folder= '.../'+elm_sym+'/'+dataset+'/WE/'
    oc_folder= '.../'+elm_sym+'/'+dataset+'/OC/'
    ae_folder= '.../'+elm_sym+'/'+dataset+'/AE/'
     
    plot_clusters(X_embedded, clusters, y_true, we_folder, 0) #0 for without elimination
    plot_clusters(X_conf_emb, clusters2, y_true[y>thr], oc_folder, 1) #1 for only confident samples
    plot_clusters(X_embedded, P, y_true, ae_folder, 2) #2 for after elimination
     
    # Concatenation of the results
    ARI_all = pd.DataFrame([all_ARI_we, all_ARI_oc, all_ARI_ae]).T
    NMI_all = pd.DataFrame([all_NMI_we, all_NMI_oc, all_NMI_ae]).T
    ACC_all = pd.DataFrame([all_ACC_we, all_ACC_oc, all_ACC_ae]).T
    
    all_results_list.append([dataset, num_class, elimination_rate, thr,
                np.mean(ARI_all[0]), np.mean(ARI_all[1]), np.mean(ARI_all[2]),
                np.mean(NMI_all[0]),np.mean(NMI_all[1]),np.mean(NMI_all[2]),
                np.mean(ACC_all[0]),np.mean(ACC_all[1]),np.mean(ACC_all[2]),
                len(np.unique(clusters)), len(np.unique(clusters2)), len(np.unique(P))])
    
    all_results_df = pd.DataFrame(all_results_list, columns=col_names)
    
    ARI_all.columns = ['without \nelimination', 'only confident \nsamples', 'after \nelimination']
    NMI_all.columns = ['without \nelimination', 'only confident \nsamples', 'after \nelimination']


# saving the dataframe
all_results_df.to_csv('.../' + elm_sym + '/DBSCAN_results.csv')
all_results_df.to_excel('.../' + elm_sym + '/DBSCAN_results.xlsx')  

#%% Run K-Means Clustering Experiments

Datasets = ['Baron_Human', 'AMB18', 'Segerstolpe', 'Baron_Mouse', 'CellBench', 'Muraro']

num_classes = [14,15,13,13,5,9]
repeat_number = 10
col_names = ['dataset','num_class','chosen_thr', 'conf_thr', 'ARI_we','ARI_oc','ARI_ae','NMI_we','NMI_oc','NMI_ae','ACC_we','ACC_oc','ACC_ae']

all_results_list = []
    
elimination_rate=0.9

if elimination_rate == 0.8:
    elm_sym = '0_8'
if elimination_rate == 0.7:
    elm_sym = '0_7' 
if elimination_rate == 0.9:
    elm_sym = '0_9'
if elimination_rate == 0.6:
    elm_sym = '0_6'
    
for dataset, num_class in zip(Datasets, num_classes):
    all_ARI_we = [] #ARI score without elimination (with whole samples)
    all_ARI_oc = [] #ARI scores with only confident values
    all_ARI_ae = [] #ARI scores after elimination
    all_NMI_we = [] #NMI score without elimination (with whole samples)
    all_NMI_oc = [] #NMI scores with only confident values
    all_NMI_ae = [] #NMI scores after elimination
    all_ACC_we = [] #ACC score without elimination (with whole samples)
    all_ACC_oc = [] #ACC scores with only confident values
    all_ACC_ae = [] #ACC scores after elimination
    
    #load dataset 
    X, y, y_true = load_data(dataset)
    thr = calculate_thr(y, elimination_rate)
    print('thr:',thr)
    # take the confident samples
    X_conf = X[y>thr]
    
    #number of components for PCA
    n_c = 30
    #apply PCA dim-red before clustering
    pca = PCA(n_components = n_c)
    
    X_embedded = pca.fit_transform(X)
    # Low-confident samples will be eliminated according to the thr value obtained.
    X_conf_emb = X_embedded[y>thr] #eliminated version of the data
    
    #best k value is chosen by elbow method
    #best k value without any elimination
    bestk = best_elbow(X_embedded)
    
    #best k value after the elimination
    bestk2 = best_elbow(X_conf_emb)
    
    for j in range(repeat_number):
        
        ###############################
        # Without Confidence Scores - Without any elimination 
        ###############################
        
        we_folder= '.../'+elm_sym+'/'+dataset+'/WE/'
        
        #apply k-means 
        kmeans = KMeans(n_clusters = bestk)
        kmeans.fit(X_embedded)
        y_kmeans = kmeans.predict(X_embedded)
        
        #calculate the evaluation metrics
        ARI_we = adjusted_rand_score(y_true, y_kmeans)
        NMI_we = normalized_mutual_info_score(y_true, y_kmeans)
        cm = confusion_matrix(y_true, y_kmeans)
        ACC_we = cal_accuracy(cm)      
        
        #append the results
        all_ACC_we.append(ACC_we)
        all_ARI_we.append(ARI_we)
        all_NMI_we.append(NMI_we)
        
        ###############################
        # With only Confident Samples
        ###############################
        
        oc_folder= '.../'+elm_sym+'/'+dataset+'/OC/'
        
        kmeans2 = KMeans(n_clusters = bestk2)
        kmeans2.fit(X_conf_emb)
        y_kmeans2 = kmeans2.predict(X_conf_emb)
        
        #calculate the evaluation metrics
        ARI_oc = adjusted_rand_score(y_true[y>thr], y_kmeans2)
        NMI_oc = normalized_mutual_info_score(y_true[y>thr], y_kmeans2)
        cm2 = confusion_matrix(y_true[y>thr], y_kmeans2)
        ACC_oc = cal_accuracy(cm2)
        
        #append the results
        all_ACC_oc.append(ACC_oc)
        all_ARI_oc.append(ARI_oc)
        all_NMI_oc.append(NMI_oc)

        ###############################
        # With Confidence Scores - After Elimination using the trained model on not eliminated data
        ###############################
        
        ae_folder= '.../'+elm_sym+'/'+dataset+'/AE/'
        
        # Predict the clusters with the trained model
        P = kmeans2.predict(X_embedded) 
        
        #calculate the evaluation metrics
        ARI_ae = adjusted_rand_score(y_true, P)
        NMI_ae = normalized_mutual_info_score(y_true, P)
        cm3 = confusion_matrix(y_true, P)
        ACC_ae = cal_accuracy(cm3)
        
        #append the results
        all_ACC_ae.append(ACC_ae)
        all_ARI_ae.append(ARI_ae)
        all_NMI_ae.append(NMI_ae)

    #visualizing the clustering results
    #plot clusters with centers
    centers = kmeans.cluster_centers_
    plot_clusters(X_embedded, centers, y_kmeans, y_true, we_folder, 0) #0 for without elimination
    centers2 = kmeans2.cluster_centers_
    plot_clusters(X_conf_emb, centers2, y_kmeans2, y_true[y>thr], oc_folder, 1) #1 for only confident samples
    plot_clusters(X_embedded, centers2, P, y_true, ae_folder, 2) #2 for after elimination
    
    # Concatenate the results
    ARI_all = pd.DataFrame([all_ARI_we, all_ARI_oc, all_ARI_ae]).T
    NMI_all = pd.DataFrame([all_NMI_we, all_NMI_oc, all_NMI_ae]).T
    ACC_all = pd.DataFrame([all_ACC_we, all_ACC_oc, all_ACC_ae]).T
    
    all_results_list.append([dataset, num_class, elimination_rate, thr,
                np.mean(ARI_all[0]), np.mean(ARI_all[1]), np.mean(ARI_all[2]),
                np.mean(NMI_all[0]),np.mean(NMI_all[1]),np.mean(NMI_all[2]),
                np.mean(ACC_all[0]),np.mean(ACC_all[1]),np.mean(ACC_all[2])])
    
    all_results_df = pd.DataFrame(all_results_list, columns=col_names)
    
    ARI_all.columns = ['without \nelimination', 'only confident \nsamples', 'after \nelimination']
    NMI_all.columns = ['without \nelimination', 'only confident \nsamples', 'after \nelimination']

# saving the dataframe
all_results_df.to_csv('.../'+elm_sym+'/k-means_results.csv')
all_results_df.to_excel('.../'+elm_sym+'/k-means_results.xlsx')  
