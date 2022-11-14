''' Dataset Cartography for Boosting Decision Trees

Paper: Improving Data Quality with Training Dynamics of Gradient Boosting Decision Trees
MA Ponti, LA Oliveira, JM Román, L Argerich

- based on:
    Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics
    Swayamdipta et al. 2020

- author:
    Moacir A. Ponti
    Mercado Livre, 2022
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#import lightgbm as lgbm
from tqdm import tqdm
from sklearn.neighbors import BallTree
import pickle
from scipy import stats

def dataset_cartography(model, X_train, y_train, label='all', method='lgbm', range_estimators=[0,100]):
    '''
    Compute Dataset Cartography for a Multiclass Model
    
    Parameters
    ----------
        model  : lgbm.LGBMClassifier, xgb.XGBClassifier
        X_train: pd.Dataframe containing the features for training
        y_train: np.array containing numeric training labels
        label  : 'all' for all classes, or TODO: a numerical label [0,n_classes] for specific class
        method : 'lgbm' (default), 'xgboost' (respective to the model)
        range_estimators: range of the iterations to compute values in percentage, default:[0,100]
                          one may not use the first iterations since the model is still not sufficiently good
                          or use a few iterations for quick tests/debug
    Returns
    -------
        tuple (confid, variab, correc):
        - confid: average confidence for each instance along estimators
        - variab: standard deviation of confidences
        - correc: instance correctness (percentage of estimators that correctly classifies the instances)
                  an instance with correc==0 is never classified correctly, at any iterations
    '''
    
    y_pred_i = []
    y_clas_i = []
    
    y_train = np.array(y_train)
    
    initial = int(model.n_estimators*(range_estimators[0]/100))
    final = int(model.n_estimators*(range_estimators[1]/100))
    
    # for each predictor
    for i in tqdm(range(initial,final)):
        # get scores and classes
        if method == 'lgbm':
            preds_ = model.predict_proba(X_train, start_iteration=initial, num_iteration=i)
        elif method == 'xgboost':
            preds_ = model.predict_proba(X_train, iteration_range=(initial,i))
        
        # compute scores and append
        scores = [s[c] for s,c in zip(preds_,y_train)] # get max score for AP
        class_ = np.argmax(preds_, axis=1)
        
        y_clas_i.append(class_)
        y_pred_i.append(scores)

    # convert to array
    y_pred_i = np.array(y_pred_i)
    y_clas_i = np.array(y_clas_i)
    
    # compute metrics (confidence, variability)
    confid = np.round(np.mean(y_pred_i, axis=0).astype(np.double), 3)
    variab = np.round(np.std(y_pred_i, axis=0).astype(np.double), 3)

    correc = (np.sum((y_clas_i==y_train), axis=0)/y_clas_i.shape[0])
    
    return confid, variab, correc
    

def plot_dataset_cartography(confid, variab, correc, y_train, dic_classes, subplot_size=(2,2)):
    '''
    Plot the dataset cartography
    Parameters
    ----------
        confid: average confidences per instance
        variab: standard deviation of confidences per instance
        correc: correctness of instances
        y_train: numeric labels of instances
        dic_classes: dictionary with name (key) and numeric label (value) in format str:int, e.g. 'negative':0
    '''
    
    fig, axs = plt.subplots(nrows=subplot_size[0], ncols=subplot_size[1], figsize=(9, 9))
    plt.subplots_adjust(hspace=0.5)
    fig.suptitle("Dataset Cartography", fontsize=14, y=0.95)
    
    for (nam,lab), ax in zip(dic_classes.items(), axs.ravel()):
        vari = variab[y_train==lab]
        conf = confid[y_train==lab]
        n_dat= len(vari)
        scat = ax.scatter(vari, conf, c=correc[y_train==lab], cmap=cm.jet)
        # chart formatting
        ax.set_title(nam+' ('+str(n_dat)+')')
        ax.set_xlabel('Variability')
        ax.set_ylabel('Confidence')
        ax.set_ylim([0, 1])

    fig.colorbar(scat, ax=axs.ravel().tolist())
    
    
def search_inconsistent_neighbors(X_train, y_train, mask_noisy, search_neighbors=1, verbose=True):
    '''
    Find noisy instances by looking at nearest neighbors of those with low correctness belonging to different class
    Uses dataset_cartography function for this task
    
    Parameters
    ----------
        X_train : pd.Dataframe containing the features for training
        y_train : pd.Dataframe or np.array containing numeric training labels
        mask_noisy: mask for noisy or pathological examples
        search_neighbors: number of neighbors to consider in the search (>=2)
    Returns
    --------
        boolean mask with the noisy instances indicated by True
    '''
    
    if search_neighbors < 1:
        raise ValueError(f'Number of neighbors must be >= 1 (have {search_neighbors})')
    
    # imput null values
    X_train_imp = np.nan_to_num(np.array(X_train), nan=0.0)
    y_train = np.array(y_train)
    
    if verbose: 
        print("Computing instances' neighbors via Ball Tree 16...")
    # compute Ball Tree
    tree = BallTree(X_train_imp, leaf_size=16) 
    
    if verbose: 
        print("Searching {} nearest neighbor(s) of {} instances...".format(search_neighbors, sum(mask_noisy)))
    # query each noisy/pathological instance for its nearest neighbors
    dist, ind = tree.query(X_train_imp[mask_noisy], k=search_neighbors+1) 
    
    print("Finding inconsistent labels in neighborhood")
    # define mask of inconsistent neighbors to be the same size of input mask
    mask_inconsist = np.array([False]*len(mask_noisy))

    # find inconsistent labels in the neighborhood
    for idx in tqdm(ind, disable=(not verbose)):
        # if only one neighbor is selected, then a simple conditional is employed
        if search_neighbors == 1:
            if y_train[idx[0]] != y_train[idx[1]]:
                mask_inconsist[idx[0]] = True
        else:
            # retrieve the labels in the neighbourhood
            labels_n = [y_train[idx[nn]] for nn in range(search_neighbors+1)]
            # find the most frequent label and check if it contradicts the noisy one
            labs, lqtd = np.unique(labels_n, return_counts=True)
            # mark as inconsistent those contradicting the most frequent one
            for it in range(search_neighbors+1):
                if labs[lqtd==np.max(lqtd)][0] != y_train[idx[it]]:
                    mask_inconsist[idx[it]] = True
            
    if verbose:
        print('Initial pathological/noisy examples:', sum(mask_noisy))
        print('Inconsistent instances found during neighbor search:', sum(mask_inconsist))

    return mask_inconsist

