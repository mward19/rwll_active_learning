#utils.py
import graphlearning as gl
import os
import numpy as np
import scipy.sparse as sparse
from copy import deepcopy
import acquisitions

from utils_representations import (
    apply_representation,
    get_base_features,
    get_representation_tag,
    representation_depends_on_seed,
    representation_is_dynamic
)


def get_models(G, model_names):
    MODELS = {'poisson':gl.ssl.poisson(G),  # poisson learning
              'laplace':gl.ssl.laplace(G), # laplace learning
              'rwll' : gl.ssl.laplace(G, reweighting='poisson'),
              'rwll1000': gl.ssl.laplace(G, reweighting='poisson', tau=0.1),
              'rwll0100': gl.ssl.laplace(G, reweighting='poisson', tau=0.01),
              'rwll0010': gl.ssl.laplace(G, reweighting='poisson', tau=0.001)
              }

    return [deepcopy(MODELS[name]) for name in model_names]

def get_model(G, model_name):
    return get_models(G, [model_name])[0]

def load_graph(dataset, metric, numeigs=200, data_dir="data", returnX=False, returnK=False, knn=0, rep_cfg=None, labeled_ind=None, seed=None, rate=1):
    # New data loading
    X_base, clusters = get_base_features(dataset, metric)

    if dataset.split("-")[-1] == 'evenodd':
        labels = clusters % 2
    elif dataset.split("-")[-1][:3] == "mod":
        modnum = int(dataset[-1])
        labels = clusters % modnum
    else:
        labels = clusters

    # Apply the transformation
    if rep_cfg is None:
        X = np.asarray(X_base, dtype=float)
        rep_tag = "regular"
    else:
        X = apply_representation(
            X_base,
            rep_cfg,
            labels=labels,
            labeled_ind=labeled_ind,
        )
        rep_tag = get_representation_tag(rep_cfg)
        if representation_depends_on_seed(rep_cfg):
            if seed is None:
                raise ValueError("Seed-dependent representations require a seed.")
            rep_tag = f"{rep_tag}_seed{seed}_r{rate}"

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
    
    # graph_filename = os.path.join(data_dir, f"{dataset.split('-')[0]}_{knn}")

    #-----
    # New file pathing
    graph_filename = os.path.join(data_dir, f"{dataset.split('-')[0]}_{metric}_{rep_tag}_{knn}")
    print(f"[GRAPH] rep tag = {rep_tag}")
    print(f"[GRAPH] knn = {knn}")
    print(f"[GRAPH] X shape = {X.shape}")
    print(f"[GRAPH] graph file = {graph_filename}")
    #-----

    normalization = "combinatorial"
    method = "lowrank"
    if dataset.split("-")[0] in ["mnist", "fashionmnist", "cifar", "emnist", "mnistsmall", "fashionmnistsmall", "salinassub", "paviasub", "mnistimb", "fashionmnistimb", "emnistvcd"]:
        normalization = "normalized"
    if labels.size < 100000:
        method = "exact"
    
    print(f"Eigendata calculation will be {method}")

    print(f"Graph Filename: {graph_filename}")

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


def build_graph_from_features(X, labels, dataset, metric, knn=0):
    """
    Build a graph directly from features in memory, with no file caching.
    Returns:
        G, trainset, normalization
    """
    if dataset.split("-")[0] == 'mstar':
        trainset = None
    elif metric == "hsi":
        trainset = np.where(labels != 0)[0]
    else:
        trainset = None

    if knn == 0:
        knn = 20
        if dataset == 'isolet':
            print("Using knn = 5 for Isolet")
            knn = 5
        elif dataset in ['box', 'blobs']:
            print(f"knn = 100, {dataset}")
            knn = 100

    normalization = "combinatorial"
    if dataset.split("-")[0] in [
        "mnist", "fashionmnist", "cifar", "emnist", "mnistsmall",
        "fashionmnistsmall", "salinassub", "paviasub", "mnistimb",
        "fashionmnistimb", "emnistvcd"
    ]:
        normalization = "normalized"

    if metric == "hsi":
        sim_name = "angular"
    else:
        sim_name = "euclidean"

    print("[DYNAMIC GRAPH] Building in-memory graph")
    print(f"[DYNAMIC GRAPH] X shape = {X.shape}")
    print(f"[DYNAMIC GRAPH] knn = {knn}")

    knn_ind, knn_dist = gl.weightmatrix.knnsearch(
        X,
        knn,
        similarity=sim_name,
        metric=metric,
        dataset=dataset.split("-")[0],
    )
    W = gl.weightmatrix.knn(X, knn, knn_data=(knn_ind, knn_dist), metric=metric)
    G = gl.graph(W)

    return G, trainset, normalization


def get_eig_data(G, normalization, numeigs):
    """
    Retrieve stored eigendata if enough is already cached on the graph;
    otherwise compute it.
    """
    eigvals = G.eigendata[normalization]['eigenvalues']
    eigvecs = G.eigendata[normalization]['eigenvectors']

    if eigvals is not None and eigvecs is not None and eigvals.size >= numeigs:
        print(f"Using previously stored {normalization} eigendata with {numeigs} evals")
        evals = eigvals[:numeigs]
        evecs = eigvecs[:, :numeigs]
    else:
        print(f"Computing {numeigs} {normalization} eigenpairs...")
        evals, evecs = G.eigen_decomp(normalization=normalization, k=numeigs)

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
        if unc_method is None:
            AL = gl.active_learning.active_learner(
                model,
                acq_func,
                labeled_ind.copy(),
                labeled_ind_labels.copy(),
            )
        else:
            AL = gl.active_learning.active_learner(
                model,
                acq_func,
                labeled_ind.copy(),
                labeled_ind_labels.copy(),
                unc_method=unc_method,
            )
    
    return AL





def get_graph_and_models(acq_funcs_names, model_names, args, rep_cfg=None, labeled_ind=None, seed=None, rate=1):
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
    G, labels, trainset, normalization, K = load_graph(args.dataset, 
                                                       args.metric, 
                                                       maxnumeigs, 
                                                       returnK=True, 
                                                       knn=args.knn, 
                                                       rep_cfg=rep_cfg,
                                                       labeled_ind=labeled_ind,
                                                       seed=seed, 
                                                       rate=rate)
    
    models = get_models(G, model_names)
    
    return models, labels, trainset, normalization,  K


def build_dynamic_graph_and_models(acq_funcs_names, model_names, args, rep_cfg, labeled_ind):
    """
    Rebuild representation, graph, and models from the CURRENT labeled set.
    No persistent graph caching is used.
    """
    X_base, clusters = get_base_features(args.dataset, args.metric)

    if args.dataset.split("-")[-1] == 'evenodd':
        labels = clusters % 2
    elif args.dataset.split("-")[-1][:3] == "mod":
        modnum = int(args.dataset[-1])
        labels = clusters % modnum
    else:
        labels = clusters

    X_rep = apply_representation(
        X_base,
        rep_cfg,
        labels=labels,
        labeled_ind=labeled_ind,
    )

    G, trainset, normalization = build_graph_from_features(
        X_rep,
        labels,
        args.dataset,
        args.metric,
        knn=args.knn,
    )

    maxnumeigs = 0
    for acq_func_name in acq_funcs_names:
        if len(acq_func_name.split("-")) == 1:
            continue
        d = acq_func_name.split("-")[-1]
        if len(d) > 0:
            if maxnumeigs < int(d):
                maxnumeigs = int(d)

    if maxnumeigs > 0:
        method = "exact" if labels.size < 100000 else "lowrank"
        print(f"[DYNAMIC GRAPH] Precomputing {maxnumeigs} eigenpairs ({normalization}, {method})")
        G.eigen_decomp(normalization=normalization, k=maxnumeigs, method=method)

    models = get_models(G, model_names)
    K = np.unique(clusters).size
    return models, labels, trainset, normalization, K