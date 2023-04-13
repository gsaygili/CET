
import numpy as np 
from extract_feat_joint_7 import extract_feats
from utils_functions import apply_tsne, find_errors_majority, find_error_score, calc_npr, evaluate_regression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
import pickle
from openpyxl import Workbook

dataset_names = ['AMB18', 'Baron_Mouse', 'Baron_Human', 'CellBench', 'Segerstolpe', 'Muraro']
data_folder = '.../'

emb_folder = ".../"

_all_ = ['cosine', 'braycurtis', 'dice', 'correlation', 'pearson','WIAD', 'kullbackleibler']

#%% Apply Embedding
#Applying t-SNE with apply_tsne function
for i, dName in enumerate(dataset_names):
    print('experiments for ' + dName + ' starts')
    x = np.load(data_folder + dataset_names [i] + "/X.npy")
    X_emb = apply_tsne(x)
    np.save(data_folder + dataset_names [i] + "/X_emb.npy",X_emb)


#%% Apply feature extraction with extract_feats function
dataset_names = ['Baron_Human']
for i, dName in enumerate(dataset_names):
    print('features for ' + dName + ' is extracting')
    emb_folder = data_folder + dataset_names [i] + "/"   
    extract_feats(emb_folder, distance_measures = _all_, K=20)
    
#%% Load dataset, train on AMB18 dataset and test on all datasets

train_data = "AMB18"
test_sets = ['AMB18', 'Baron_Mouse', 'Baron_Human', 'CellBench', 'Segerstolpe', 'Muraro'] 
load_folder = data_folder + train_data 

Data = np.load(load_folder + 'X.npy')
Emb_Data = np.load(load_folder + 'X_emb.npy')
Label = np.load(load_folder + 'y.npy', allow_pickle=True)
Feat = np.load(load_folder + 'X_feat.npy')

y = Label.ravel()
le = preprocessing.LabelEncoder()
Label = le.fit_transform(y)

X_train, X_test, y_tr, y_test = train_test_split(Data, Label, stratify=Label, test_size=0.2, random_state=42)
X_tr_emb, X_test_emb, y_tr, y_test = train_test_split(Emb_Data, Label, stratify=Label, test_size=0.2, random_state=42)
X_train_feat, X_test_feat, y_tr, y_test = train_test_split(Feat, Label, stratify=Label, test_size=0.2, random_state=42)


np.save(load_folder+"X_train_feat.npy", X_train_feat)
np.save(load_folder+"X_test_feat.npy", X_test_feat)

#reshaping feature data
X_train_feat = np.reshape(X_train_feat, [X_train_feat.shape[0], X_train_feat.shape[1]*X_train_feat.shape[2]])

np.save(load_folder+"X_train.npy", X_train)
np.save(load_folder+"X_test.npy", X_test)
np.save(load_folder+"y_train.npy", y_tr)
np.save(load_folder+"y_test.npy", y_test)
np.save(load_folder+"X_train_emb.npy", X_tr_emb)
np.save(load_folder+"X_test_emb.npy", X_test_emb)

#calculate error scores for training set(y_tr_score), y_tr_ind--indexes of erroneous samples, y_tra--binary labels (erroneous-0,corrects-1) 
y_tr_ind, y_tr_score = find_error_score(X_tr_emb, y_tr)
y_tra = np.ones(X_tr_emb.shape[0])
y_tra[y_tr_ind] = 0

y_tr_score= np.array(y_tr_score)
#X_train=np.reshape(x_train, [x_train.shape[0],x_train.shape[1]*x_train.shape[2]])
print("X_train_feat shape:" , X_train_feat.shape)
print("y_tr_score shape:" , y_tr_score.shape)

#calculate error scores for test set(y_te_score), y_te_ind--indexes of erroneous samples, y_tes--binary labels (erroneous-0,corrects-1)
y_te_ind, y_te_score = find_error_score(X_test_emb, y_test)
y_tes = np.ones(X_test_emb.shape[0])
y_tes[y_te_ind] = 0

#y_tes=np.reshape(y_tes, [y_tes.shape[0],1])
y_te_score=np.array(y_te_score)
print("X_test_feat shape:" , X_test_feat.shape)
print("y_te_score shape:" , y_te_score.shape)
np.save(load_folder+"y_te_score.npy", y_te_score)
#calculate NPR for test set
y_npr = calc_npr(X_test, X_test_emb)

#Train the model
print("training of the model")

param_grid = {
    'n_estimators': [20, 50, 100, 200],
    'max_features': ["auto", "sqrt"],
    'criterion': ["squared_error"],
    'max_depth': [2, 5, 10, 20],
    'min_samples_split': [2, 5, 10, 20],
}

grid = GridSearchCV(RandomForestRegressor(random_state = 42),
                    param_grid, refit=True, verbose = 0, cv = 3)
grid.fit(X_train_feat, y_tr_score)

#best model
clf = grid.best_estimator_

path_model = ".../best_RF_model_for_AMB18.sav"
#save model
pickle.dump(clf, open(path_model, 'wb'))
print('completed')

#EVALUATION PART for training on AMB18

save_folder = '.../Results_on_AMB18/'
clf = pickle.load(open(path_model, 'rb'))

wb = Workbook()
sheet = wb.active  
sheet['A1'] = "Training Dataset"
sheet["B1"] = "Test Dataset"
sheet['C1'] = "Total error"
sheet['D1'] = "100"
sheet['E1'] = "50"
sheet['F1'] = "10"
sheet['G1'] = "100"
sheet['H1'] = "50"
sheet['I1'] = "10"
wb.save(save_folder + "exp_results.xlsx")


for i, testName in enumerate(test_sets):
    print('experiments for ' + testName + ' starts')
    
    load_folder = data_folder + testName +'/UMAP/'
    
    if (testName == 'AMB18' or "Baron_Human"):
        X_test = np.load(load_folder + 'X_test.npy')
        y_test = np.load(load_folder + 'y_test.npy', allow_pickle=True)
        X_test_emb = np.load(load_folder + 'X_test_emb.npy')
        X_test_feat = np.load(load_folder + 'X_test_feat.npy')
        X_test_feat = np.reshape(X_test_feat, [X_test_feat.shape[0], X_test_feat.shape[1]*X_test_feat.shape[2]])
    
    else:
        X_test = np.load(load_folder + 'X.npy')
        y_test = np.load(load_folder + 'y.npy', allow_pickle=True)
        X_test_emb = np.load(load_folder + 'X_emb.npy')
        X_test_feat = np.load(load_folder + 'X_feat.npy')
        X_test_feat = np.reshape(X_test_feat, [X_test_feat.shape[0], X_test_feat.shape[1]*X_test_feat.shape[2]])
        
    #calculate error scores for test set(y_te_score), y_te_ind--indexes of erroneous samples, y_tes--binary labels (erroneous-0,corrects-1)
    y_te_ind, y_te_score = find_error_score(X_test_emb, y_test)
    y_tes = np.ones(X_test_emb.shape[0])
    y_tes[y_te_ind] = 0

    #y_tes=np.reshape(y_tes, [y_tes.shape[0],1])
    y_te_score=np.array(y_te_score)

    #calculate NPR for test set
    y_npr = calc_npr(X_test, X_test_emb)
    
    #evaluate_regression(clf, X_test_feat, y_te_score, y_te_ind, y_npr, save_folder)
    evaluate_regression(clf, X_test_feat, y_te_score, y_te_ind, y_npr, save_folder, train_data, testName)
    
       
    
#%% Load dataset, train on BH dataset and test on all datasets

train_data = "Baron_Human"
test_sets = ['AMB18', 'Baron_Mouse', 'Baron_Human', 'CellBench', 'Segerstolpe', 'Muraro'] 
load_folder = data_folder + train_data

Data = np.load(load_folder + 'X.npy')
Emb_Data = np.load(load_folder + 'X_emb.npy')
Label = np.load(load_folder + 'y.npy', allow_pickle=True)
Feat = np.load(load_folder + 'X_feat.npy')

y = Label.ravel()
le = preprocessing.LabelEncoder()
Label = le.fit_transform(y)

X_train, X_test, y_tr, y_test = train_test_split(Data, Label, stratify=Label, test_size=0.2, random_state=42)
X_tr_emb, X_test_emb, y_tr, y_test = train_test_split(Emb_Data, Label, stratify=Label, test_size=0.2, random_state=42)
X_train_feat, X_test_feat, y_tr, y_test = train_test_split(Feat, Label, stratify=Label, test_size=0.2, random_state=42)


np.save(load_folder+"X_train_feat.npy", X_train_feat)
np.save(load_folder+"X_test_feat.npy", X_test_feat)

#reshaping feature data
X_train_feat = np.reshape(X_train_feat, [X_train_feat.shape[0], X_train_feat.shape[1]*X_train_feat.shape[2]])
#X_test_feat = np.reshape(X_test_feat, [X_test_feat.shape[0], X_test_feat.shape[1]*X_test_feat.shape[2]])


np.save(load_folder+"X_train.npy", X_train)
np.save(load_folder+"X_test.npy", X_test)
np.save(load_folder+"y_train.npy", y_tr)
np.save(load_folder+"y_test.npy", y_test)
np.save(load_folder+"X_train_emb.npy", X_tr_emb)
np.save(load_folder+"X_test_emb.npy", X_test_emb)


#calculate error scores for training set(y_tr_score), y_tr_ind--indexes of erroneous samples, y_tra--binary labels (erroneous-0,corrects-1) 
y_tr_ind, y_tr_score = find_error_score(X_tr_emb, y_tr)
y_tra = np.ones(X_tr_emb.shape[0])
y_tra[y_tr_ind] = 0

y_tr_score= np.array(y_tr_score)
#X_train=np.reshape(x_train, [x_train.shape[0],x_train.shape[1]*x_train.shape[2]])
print("X_train_feat shape:" , X_train_feat.shape)
print("y_tr_score shape:" , y_tr_score.shape)

#calculate error scores for test set(y_te_score), y_te_ind--indexes of erroneous samples, y_tes--binary labels (erroneous-0,corrects-1)
y_te_ind, y_te_score = find_error_score(X_test_emb, y_test)
y_tes = np.ones(X_test_emb.shape[0])
y_tes[y_te_ind] = 0

#y_tes=np.reshape(y_tes, [y_tes.shape[0],1])
y_te_score=np.array(y_te_score)
print("X_test_feat shape:" , X_test_feat.shape)
print("y_te_score shape:" , y_te_score.shape)
np.save(load_folder+"y_te_score.npy", y_te_score)
#calculate NPR for test set
y_npr = calc_npr(X_test, X_test_emb)

#Train the model
print("training of the model")

param_grid = {
    'n_estimators': [20, 50, 100, 200],
    'max_features': ["auto", "sqrt"],
    'criterion': ["squared_error"],
    'max_depth': [2, 5, 10, 20],
    'min_samples_split': [2, 5, 10, 20],
}

grid = GridSearchCV(RandomForestRegressor(random_state = 42),
                    param_grid, refit=True, verbose = 0, cv = 3)
grid.fit(X_train_feat, y_tr_score)

#best model
clf = grid.best_estimator_

path_model = ".../best_RF_model_for_BH.sav"
#save model
pickle.dump(clf, open(path_model, 'wb'))
print('completed')

#EVALUATION PART for training on Baron Human
save_folder = '.../Results_on_Baron_Human/'
clf = pickle.load(open(path_model, 'rb'))

wb = Workbook()
sheet = wb.active  
sheet['A1'] = "Training Dataset"
sheet["B1"] = "Test Dataset"
sheet['C1'] = "Total error"
sheet['D1'] = "100"
sheet['E1'] = "50"
sheet['F1'] = "10"
sheet['G1'] = "100"
sheet['H1'] = "50"
sheet['I1'] = "10"
wb.save(save_folder + "exp_results.xlsx")

for i, testName in enumerate(test_sets):
    print('experiments for ' + testName + ' starts')
    
    load_folder = data_folder + testName +'/UMAP/'
    
    if (testName == 'AMB18' or "Baron_Human"):
        X_test = np.load(load_folder + 'X_test.npy')
        y_test = np.load(load_folder + 'y_test.npy', allow_pickle=True)
        X_test_emb = np.load(load_folder + 'X_test_emb.npy')
        X_test_feat = np.load(load_folder + 'X_test_feat.npy')
        X_test_feat = np.reshape(X_test_feat, [X_test_feat.shape[0], X_test_feat.shape[1]*X_test_feat.shape[2]])
           
    else:
        X_test = np.load(load_folder + 'X.npy')
        y_test = np.load(load_folder + 'y.npy', allow_pickle=True)
        X_test_emb = np.load(load_folder + 'X_emb.npy')
        X_test_feat = np.load(load_folder + 'X_feat.npy')
        X_test_feat = np.reshape(X_test_feat, [X_test_feat.shape[0], X_test_feat.shape[1]*X_test_feat.shape[2]])
        
    #calculate error scores for test set(y_te_score), y_te_ind--indexes of erroneous samples, y_tes--binary labels (erroneous-0,corrects-1)
    y_te_ind, y_te_score = find_error_score(X_test_emb, y_test)
    y_tes = np.ones(X_test_emb.shape[0])
    y_tes[y_te_ind] = 0

    #y_tes=np.reshape(y_tes, [y_tes.shape[0],1])
    y_te_score=np.array(y_te_score)

    #calculate NPR for test set
    y_npr = calc_npr(X_test, X_test_emb)
    
    #evaluate_regression(clf, X_test_feat, y_te_score, y_te_ind, y_npr, save_folder)
    evaluate_regression(clf, X_test_feat, y_te_score, y_te_ind, y_npr, save_folder, train_data, testName)
    
       
    
