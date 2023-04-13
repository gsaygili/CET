

"""
This file contains utility functions for obtaining t-SNE embeddings, detecting erroneously 
    embedded samples, calculating ground-truth confidence scores, and 
    evaluating confidence estimation model.

Functions:
----------

def apply_tsne(data, dim=2, perplexity=30)
    Perform t-SNE dimensionality reduction on a given dataset and 
    returns the embedded data.

    Input:
        data (numpy.ndarray): A 2D or 3D numpy array representing the input data. If it is 3D, then it should be reshaped.
        dim (int, optional): The dimensionality of the output data. Defaults to 2.
        perplexity (int, optional): The perplexity hyperparameter for t-SNE. Defaults to 30.

    Output:
        numpy.ndarray: A numpy array representing the embedded data.

find_errors_majority(X_emb, labels, K=20)
    Find the indices of erroneously embedded samples whose labels do not match the majority of their K nearest neighbors.
    
    Input:
    - X_emb: array-like, shape (n_samples, n_features)
        Embedding matrix of samples.
    - labels: array-like, shape (n_samples,)
        Labels of samples.
    - K: int, optional (default=20)
        Number of nearest neighbors to consider.
    
    Output:
    - err: array-like, shape (n_errors,)
        Indices of samples that have a different majority label than their K-nearest neighbors.

find_error_score(X_emb, labels, K=20)
    Find the indices of samples whose labels do not match the majority of their K nearest neighbors, and also calculate the confidence score (proportion of their K nearest neighbors with the same label).
    
    Input:
    - X_emb: array-like, shape (n_samples, n_features)
        Embedding matrix of samples.
    - labels: array-like, shape (n_samples,)
        Labels of samples.
    - K: int, optional (default=20)
        Number of nearest neighbors to consider.
        
    Output:
    - err: array-like, shape (n_errors,)
        Indices of samples that have a different majority label than their K-nearest neighbors.
    - Score: array-like, shape (n_samples,)
        Confidence score per sample
calc_npr(X, X_emb, K=20)
    Calculate the Neighborhood Preservation Ratio (NPR) for each sample.
    
    Input:
    - X: array-like, shape (n_samples, n_features)
        Data matrix of samples.
    - X_emb: array-like, shape (n_samples, n_embedding_features)
        Embedding matrix of samples.
    - K: int, optional (default=20)
        Number of nearest neighbors to consider.

    Output:
    - npr: array-like, shape (n_samples,)
        Neighborhood preservation ratio for each sample.
    

evaluate_regression(model, X_test, y_te_score, y_te_ind, y_npr, save_folder, tr_data, test_data)
    Evaluate the performance of the RF regression model.
    
    Input:
    - model: scikit-learn regressor object
        The fitted regression model.
    - X_test: array-like, shape (n_test_samples, n_features)
        Test set feature matrix.
    - y_te_score: array-like, shape (n_test_samples,)
        Ground-truth confidence scores for test set.
    - y_te_ind: array-like, shape (n_test_samples,)
        Indexes of erroneously embedded samples
    - y_npr: array-like, shape (n_samples,)
        NPR values for test set
    - save_folder: str
        Path to the folder where evaluation logs and figures will be saved.
    - tr_data: str
        Name of the training data. (just to summarise results appropriately)
    - test_data: str
        Name of the test data. (just to summarise results appropriately)
        
    Output:
        saved results
        
"""

import numpy as np
import sys
import scipy.spatial.distance as dist
from scipy.stats import variation
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from openpyxl import load_workbook
from sklearn.manifold import TSNE


def apply_tsne(data, dim=2, perplexity=30):
#    X = data.reshape((data.shape[0], data.shape[1]*data.shape[2]))
    X_embedded = TSNE(n_components=dim, perplexity=perplexity).fit_transform(data)
    return X_embedded


# functions for detecting erroneously embedded samples and calculating ground-truth confidence scores
def find_errors_majority(X_emb, labels, K=20):
    # take a closest neighborhood of K and check whether the label of the majority is the same as the center
    X_d = dist.squareform(dist.pdist(X_emb, "euclidean"))
    # the first column is the sample itself since the distance is zero
    sort_index = np.argsort(X_d)
    error_list = []
    for i in range(X_d.shape[0]):
        K_neigh = sort_index[i, 1:K+1]
        lab_neigh = labels[K_neigh]
        num_label = np.count_nonzero(lab_neigh == labels[i])
        if num_label < K/2:
            error_list.append(i)
    err = np.array(error_list)
    return err

def find_error_score(X_emb,labels, K=20):
    # take a closest neighborhood of K and check whether the label of the majority is the same as the center
    X_d = dist.squareform(dist.pdist(X_emb, "euclidean"))
    # the first column is the sample itself since the distance is zero
    sort_index = np.argsort(X_d)
    error_list = []
    Score = []
    for i in range(X_d.shape[0]):
        K_neigh = sort_index[i, 1:K+1]
        lab_neigh = labels[K_neigh]
        num_label = np.count_nonzero(lab_neigh == labels[i])
        if num_label < K/2:
            error_list.append(i)
        score=num_label/K
        Score.append(score)
    err = np.array(error_list)
    return err,Score
    
def calc_npr(X, X_emb, K=20):
    X_d = dist.squareform(dist.pdist(X_emb, "euclidean"))
    X_D = dist.squareform(dist.pdist(X, "euclidean"))
    ind_d = np.argsort(X_d)
    ind_D = np.argsort(X_D)
    npr = np.zeros(X_d.shape[0],)
    for i in range(X_d.shape[0]):
        K_d = ind_d[i, 1:K+1]
        K_D = ind_D[i, 1:K+1]
        inter = np.intersect1d(K_d, K_D)
        count = inter.shape[0]
        npr[i] = count/K
    return npr
    
#EVALUATION PART
  
def evaluate_regression (model, X_test, y_te_score, y_te_ind, y_npr, save_folder, tr_data, test_data):
       
  y_pred = model.predict(X_test)
  
  ind_err = [ n for n,i in enumerate(y_te_score) if i<0.5] 
  
  #for writing log screen in a file
  stdoutOrigin = sys.stdout 
  sys.stdout = open(save_folder + test_data + "/log.txt", "w")

  # Calculate the absolute errors
  errors = np.abs(y_pred - y_te_score)

  print('y_pred_mean:',np.mean(y_pred))
  print('y_pred_std:',np.std(y_pred))

  print('y_test_mean:',np.mean(y_te_score))
  print('y_test_std:',np.std(y_te_score))

  # Print out the mean absolute error (mae)
  print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

  # Calculate mean squared error (MSE)
  mse=mean_squared_error(y_te_score, y_pred)
  print('Mean Squared Error:', mse)
  rmse = np.sqrt(mse)
  print('Root Mean Squared Error:', rmse)  
  print('coefficient of variation (CV):', variation(y_pred))
  
  sort_index=np.argsort(y_pred,axis=0)
  y_pred_sorted=y_pred[(sort_index)]
  sorted_test=y_te_score[(sort_index)]
  
  #calculate slop and r2
  model = LinearRegression()
  model.fit(sorted_test.reshape((-1, 1)),y_pred_sorted.reshape((-1, 1)))
  r_sq = model.score(sorted_test.reshape((-1, 1)),y_pred_sorted.reshape((-1, 1)))
  print('coefficient of determination (R2) for pred:', r_sq)
  print('slope for pred:', model.coef_)

  #Correct prediction of last N error
  print("-----for prediction-------")
  
  ind = np.argsort(y_pred)

  print('corrects in last'+ str(len(y_te_ind)) +' sample:', len(np.intersect1d(ind[:len(y_te_ind)], y_te_ind)))
  print('corrects in last 100 sample:', len(np.intersect1d(ind[:100], y_te_ind)))
  print('corrects in last 50 sample:', len(np.intersect1d(ind[:50], y_te_ind)))
  print('corrects in last 10 sample:', len(np.intersect1d(ind[:10], y_te_ind)))
  
  #for npr
  print("-----for npr-------")
  
  ind2 = np.argsort(y_npr)

  print('corrects in last'+ str(len(y_te_ind)) +' sample:', len(np.intersect1d(ind2[:len(y_te_ind)], y_te_ind)))
  print('corrects in last 100 sample:', len(np.intersect1d(ind2[:100], y_te_ind)))
  print('corrects in last 50 sample:', len(np.intersect1d(ind2[:50], y_te_ind)))
  print('corrects in last 10 sample:', len(np.intersect1d(ind2[:10], y_te_ind)))
  
  np.save(save_folder + "y_npr.npy", y_npr)
  np.save(save_folder + test_data + "/y_pred.npy", y_pred)
  
  sys.stdout.flush()
  sys.stderr.flush() 
  sys.stdout.close()
  sys.stdout=stdoutOrigin
  
  wb = load_workbook(save_folder + 'exp_results.xlsx')
  sheet = wb.active
  data = (tr_data, test_data, len(ind_err), len(np.intersect1d(ind[:100], y_te_ind)),
          len(np.intersect1d(ind[:50], y_te_ind)), len(np.intersect1d(ind[:10], y_te_ind)),
          len(np.intersect1d(ind2[:100], y_te_ind)),len(np.intersect1d(ind2[:50], y_te_ind)), 
          len(np.intersect1d(ind2[:10], y_te_ind)))
  
  sheet.append(data)
  wb.save(save_folder + 'exp_results.xlsx')

