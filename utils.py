import graphlearning as gl
import os
from collections import namedtuple
import numpy as np
import scipy.sparse as sparse
import torch
from copy import deepcopy
import acquisitions


def get_models(G, model_names):
    MODELS = {'poisson':gl.ssl.poisson(G),  # poisson learning
              'laplace':gl.ssl.laplace(G), # laplace learning
              'rwll' : gl.ssl.laplace(G, reweighting='poisson'),
              'rwll1000': gl.ssl.laplace(G, reweighting='poisson', tau=0.1),
              'rwll0100': gl.ssl.laplace(G, reweighting='poisson', tau=0.01),
              'rwll0010': gl.ssl.laplace(G, reweighting='poisson', tau=0.001)
              }

    return [deepcopy(MODELS[name]) for name in model_names]

def new_get_model(G, model_name):
    # NOTE: no need for a deep copy because we're making it on the spot
    if model_name == 'poisson':    return gl.ssl.poisson(G),  # poisson learning
    elif model_name == 'laplace':  return gl.ssl.laplace(G), # laplace learning
    elif model_name == 'rwll':     return gl.ssl.laplace(G, reweighting='poisson'),
    elif model_name == 'rwll1000': return gl.ssl.laplace(G, reweighting='poisson', tau=0.1),
    elif model_name == 'rwll0100': return gl.ssl.laplace(G, reweighting='poisson', tau=0.01),
    elif model_name == 'rwll0010': return gl.ssl.laplace(G, reweighting='poisson', tau=0.001)
    else:                          raise ValueError('Model name not found!')

def new_load_graph(
        X: np.ndarray, # The embeddings of the dataset derived from a neural network
        dataset_name: str,
        embeddings_info: namedtuple, # A named tuple (or dict, if you'd like?) containing the necessary information to determine the graph filename. NN name, epoch number, etc.
        numeigs=200,
        data_dir="data",
        knn=0, # The number of clusters to use in KNN
    ):
    """
    Using embeddings of the data (derived from a neural network in training),
    construct a graph of the dataset.
    """
    # Construct the similarity graph
    print(f"Constructing similarity graph for {dataset_name}")
    if knn == 0:
        knn = 20
        if dataset_name == 'isolet':
            print("Using knn = 5 for Isolet")
            knn = 5
        elif dataset_name in ['box', 'blobs']:
            print(f"knn = 100, {dataset_name}")
            knn = 100
    
    # TODO: Make the name involve the new NN and the epoch or something
    embeddings_name = 'placeholder' # TODO: use embeddings_info to make a filename modifier, like f'{model_name}_{epoch_number}' # 
    graph_filename = os.path.join(data_dir, f"{dataset_name.split('-')[0]}_{embeddings_name}_{knn}")

    # NOTE: what does this do?? haha
    normalization = "combinatorial"
    method = "lowrank"
    if dataset_name.split("-")[0] in ["mnist", "fashionmnist", "cifar", "emnist", "mnistsmall", "fashionmnistsmall", "salinassub", "paviasub", "mnistimb", "fashionmnistimb", "emnistvcd"]:
        normalization = "normalized"
    # NOTE: this used to be labels.size, we changed it because in load_graph,
    # `labels` contains labels for the *whole* dataset, so len(X) ought to be
    # the same number
    if len(X) < 100000: 
        method = "exact"

    # NOTE: commented this out since otherwise it will print way too many times
    # print(f"Eigendata calculation will be {method}")

    try:
        G = gl.graph.load(graph_filename)
        found = True
    except:
        # if metric == "hsi":
        #     sim_name ="angular" # LAND does 100 in HSI
        # else:
        # NOTE: We (Matthew and Garrett) are not planning on working with hyperspectral imaging (hsi?) so just use the euclidean metric for knn.
        sim_name = "euclidean"
        knn_ind, knn_dist = gl.weightmatrix.knnsearch(X, knn, similarity=sim_name, metric=embeddings_name, dataset=dataset_name.split("-")[0])
        W = gl.weightmatrix.knn(X, knn, knn_data=(knn_ind, knn_dist), metric=embeddings_name)
        G = gl.graph(W)
        found = False

    if numeigs is not None:
        eigdata = G.eigendata[normalization]['eigenvalues']
        if eigdata is not None:
            prev_numeigs = eigdata.size
            if prev_numeigs >= numeigs:
                print("Retrieving Eigendata...")
            else:
                print(f"Requested {numeigs} eigenvalues, but have only {prev_numeigs} stored. Recomputing Eigendata...")
        else:
            print(f"No Eigendata found, so computing {numeigs} eigenvectors...")

        evals, evecs = G.eigen_decomp(normalization=normalization, k=numeigs, method=method)

    G.save(graph_filename)
    
    # NOTE: we do not return the number of unique labels because this should be
    # known upon loading the datas, which no longer happens here 
    # if returnK:
    #     return G, normalization, np.unique(clusters).size
    
    return G, normalization


def load_graph(
        dataset: str, 
        metric: str, 
        numeigs=200, 
        data_dir="data", 
        returnX=False, 
        returnK=False, 
        knn=0 
    ):
    """
    Get a graph of the dataset to do active learning with.

    dataset: The desired dataset's name
    metric: What kind of data to load from the dataset (raw data or embeddings of some kind)
    numeigs: Seem to be unused (results from eigen decomp not returned)
    returnX: X represents data points. Each row is a datapoint
    returnK: K is number of unique labels (clusters)
    knn: Number of nearest neighbors to use in graph construction. Lower `knn` means sparser graph

    Returns:
        G (gl.graph): the graph of the dataset
        labels (np.ndarray): the labels of the data
        trainset: labels of train data. Only used if doing a specific train/test split (?)
        normalization (str): Unused right now I think. For the eigenvalue decomposition
        X or K (optional): Either the data points or the unique cluster indices, depending on returnX or returnK
    """
    X, clusters = gl.datasets.load(dataset.split("-")[0], metric=metric)
    if dataset.split("-")[-1] == 'evenodd':
        labels = clusters % 2
    elif dataset.split("-")[-1][:3] == "mod":
        modnum = int(dataset[-1])
        labels = clusters % modnum
    else:
        labels = clusters

    if dataset.split("-")[0] == 'mstar': # allows for specific train/test set split TODO
        trainset = None
    elif metric == "hsi":
        trainset = np.where(labels != 0)[0]
    else:
        trainset = None

    # Construct the similarity graph
    print(f"Constructing similarity graph for {dataset}")
    if knn == 0:
        knn = 20
        if dataset == 'isolet':
            print("Using knn = 5 for Isolet")
            knn = 5
        elif dataset in ['box', 'blobs']:
            print(f"knn = 100, {dataset}")
            knn = 100
    
    graph_filename = os.path.join(data_dir, f"{dataset.split('-')[0]}_{knn}")

    normalization = "combinatorial"
    method = "lowrank"
    if dataset.split("-")[0] in ["mnist", "fashionmnist", "cifar", "emnist", "mnistsmall", "fashionmnistsmall", "salinassub", "paviasub", "mnistimb", "fashionmnistimb", "emnistvcd"]:
        normalization = "normalized"
    if labels.size < 100000:
        method = "exact"
    
    print(f"Eigendata calculation will be {method}")

    try:
        G = gl.graph.load(graph_filename)
        found = True
    except:
        if metric == "hsi":
            sim_name ="angular" # LAND does 100 in HSI
        else:
            sim_name = "euclidean"
        knn_ind, knn_dist = gl.weightmatrix.knnsearch(X, knn, similarity=sim_name, metric=metric, dataset=dataset.split("-")[0])
        W = gl.weightmatrix.knn(X, knn, knn_data=(knn_ind, knn_dist), metric=metric)
        G = gl.graph(W)
        found = False

    if numeigs is not None:
        eigdata = G.eigendata[normalization]['eigenvalues']
        if eigdata is not None:
            prev_numeigs = eigdata.size
            if prev_numeigs >= numeigs:
                print("Retrieving Eigendata...")
            else:
                print(f"Requested {numeigs} eigenvalues, but have only {prev_numeigs} stored. Recomputing Eigendata...")
        else:
            print(f"No Eigendata found, so computing {numeigs} eigenvectors...")

        evals, evecs = G.eigen_decomp(normalization=normalization, k=numeigs, method=method)

    G.save(graph_filename)
    
    if returnX:
        return G, labels, trainset, normalization, X
    
    if returnK:
        return G, labels, trainset, normalization, np.unique(clusters).size
    
    return G, labels, trainset, normalization


def get_eig_data(G, normalization, numeigs):
    # determine if need to recompute eigenvalues/vectors
    recompute = True
    if G.eigendata[normalization]['eigenvalues'] is not None:
        if G.eigendata[normalization]['eigenvalues'].size < numeigs:
            recompute = True
    else:
        recompute = True
        
    if not recompute:
        # Current gl.active_learning is implemented only to allow for exact eigendata compute for "normalized"
        print(f"Using previously stored {normalization} eigendata with {numeigs} evals for {acq_func_name}")
        evals = G.eigendata[normalization]['eigenvalues'][:numeigs]
        evecs = G.eigendata[normalization]['eigenvectors'][:,:numeigs] 
        
    else:
        print("Warning: Computing eigendata with gl.active_learning defaults...")
        evals, evecs = G.eigen_decomp(normalization, k=numeigs)

    return evals, evecs


def get_unc_acq_func(af_name):
    unc_method = None
    if af_name == "random":
        acq_func = acquisitions.random
    elif af_name in ["uncnorm", "uncnormdecaytau", "uncnormkde", "uncnormdecaytaukde"]:
        acq_func = gl.active_learning.unc_sampling
        unc_method = "unc_2norm"
    elif af_name == "unc":
        acq_func = gl.active_learning.unc_sampling
        unc_method = "smallest_margin"
    elif af_name in ["uncnormprop++","uncnormprop++decaytau"]:
        acq_func = acquisitions.uncnormprop_plusplus
    else:
        raise NotImplementedError(f"Acquisition function = {af_name} not implemented...")

    return acq_func, unc_method
    
def get_active_learner(acq_func_name, model, labeled_ind, labeled_ind_labels, normalization, args, numeigs=100):
    """
        Based on the acquisition function name, determine if need to compute the covariance matrix for instantiating 
        active_learner object
    """
    if len(acq_func_name.split("-")) > 1:
        numeigs = int(acq_func_name.split("-")[-1])
        
    af_name = acq_func_name.split("-")[0]
    if af_name in ["mc", "mcvopt", "vopt", "vopt1", "sopt"]:
        print(f"gamma = {args.gamma}")
        evals, V = get_eig_data(model.graph, normalization, numeigs)
        if acq_func_name.split("-")[0][-1] == "1":
            evals = evals[1:] 
            V = V[:,1:]
        C = np.linalg.inv(np.diag(evals + 1e-11))

        if af_name == "mc":
            acq_func = gl.active_learning.model_change
        elif af_name == "mcvopt":
            acq_func = gl.active_learning.model_change_var_opt
        elif af_name in ["vopt", "vopt1"]:
            acq_func = gl.active_learning.var_opt
        elif af_name == "sopt":
            acq_func = gl.active_learning.sigma_opt

        AL = gl.active_learning.active_learner(model, acq_func, labeled_ind.copy(), labeled_ind_labels.copy(), C=C.copy(), V=V.copy(), gamma2=args.gamma**2.)
        
    elif af_name in ["voptfull", "soptfull"]:
        # add a small diagonal term to make C invertible
        C = sparse.linalg.inv(sparse.csc_matrix(model.graph.laplacian(normalization=normalization) + 
                                                                       0.001*sparse.eye(model.graph.num_nodes))).toarray() 
        if af_name == "voptfull":
            acq_func = gl.active_learning.var_opt
        else:
            acq_func = gl.active_learning.sigma_opt
            
        AL = gl.active_learning.active_learner(model, acq_func, labeled_ind, labeled_ind_labels, C=C.copy(), gamma2=args.gamma**2.)

    else:
        acq_func, unc_method = get_unc_acq_func(af_name)
        AL = gl.active_learning.active_learner(model, acq_func, labeled_ind.copy(), labeled_ind_labels.copy())
    
    return AL





def get_graph_and_models(acq_funcs_names, model_names, args):
    # Determine if we need to calculate more eigenvectors/values for mc, vopt, mcvopt acquisitions
    maxnumeigs = 0
    for acq_func_name in acq_funcs_names:
#         if acq_func_name in ["mc", "mcvopt", "vopt", "vopt1"]:
#             maxnumeigs = max(maxnumeigs, 50)
            
        if len(acq_func_name.split("-")) == 1:
            continue
        d = acq_func_name.split("-")[-1]
        if len(d) > 0:
            if maxnumeigs < int(d):
                maxnumeigs = int(d)
    if maxnumeigs == 0:
        maxnumeigs = None

    # Load in the graph and labels
    print("Loading in Graph...")
    G, labels, trainset, normalization, K = load_graph(args.dataset, args.metric, maxnumeigs, returnK=True, knn=args.knn)
    
    models = get_models(G, model_names) 
    
    return models, labels, trainset, normalization,  K




def new_get_graph_and_model(model_name, args, embeddings): #NOTE: this function should be called every x number of iterations to get the new graph using the updated embeddings

    G, normalization = new_load_graph(embeddings, returnK=True, knn=args.knn)

    model = new_get_model(G, model_name)

    return model, normalization
