#test_al_gl.py
import numpy as np
import matplotlib.pyplot as plt
import graphlearning as gl
import scipy.sparse as sparse
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
import pickle
import os
import yaml
from copy import deepcopy
from glob import glob
from scipy.special import softmax
from functools import reduce
from utils import *

from utils_representations import (
    get_representation_config,
    get_representation_tag,
    representation_depends_on_seed,
    representation_is_dynamic,
    get_nn_update_interval,
)


from joblib import Parallel, delayed

if __name__ == "__main__":
    parser = ArgumentParser(description="Run Large Tests in Parallel of Active Learning Test for Graph Learning")
    parser.add_argument("--dataset", type=str, default='mnist-mod3')
    parser.add_argument("--metric", type=str, default='vae')
    parser.add_argument("--numcores", type=int, default=5)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--resultsdir", type=str, default="results")
    parser.add_argument("--config", type=str, default="./config.yaml")
    parser.add_argument("--K", type=int, default=0)
    parser.add_argument("--cheatK", type=int, default=50)
    parser.add_argument("--knn", type=int, default=0)
    args = parser.parse_args()

    # load in configuration file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    
    #-----
    # Get the representation tag for the get_features function
    rep_cfg = get_representation_config(config)
    rep_tag = get_representation_tag(rep_cfg)
    dynamic_rep = representation_is_dynamic(rep_cfg)
    nn_update_interval = get_nn_update_interval(rep_cfg)
    print("Dynamic representation:", dynamic_rep)
    if dynamic_rep:
        print("NN update interval:", nn_update_interval)
    print("Using representation:", rep_tag)
    #-----


    # Define ssl models and acquisition functions from configuration file 
    ACQS_MODELS = [name for name in config["acqs_models"] if name.split(" ")[-1][:4] != "LAND"]
    acq_funcs_names = [name.split(" ")[0] for name in ACQS_MODELS]
    

    # load in graph and models that will be used in this run of tests
    model_names = [name.split(" ")[1] for name in ACQS_MODELS]

    # Static seed-independent reps can be built once.
    # Static/dynamic NN reps are built inside the seed loop after labeled_ind is known.
    if not representation_depends_on_seed(rep_cfg):
        models, labels, trainset, normalization, K = get_graph_and_models(
            acq_funcs_names,
            model_names,
            args,
            rep_cfg=rep_cfg,
        )
    else:
        models = None
        labels = None
        trainset = None
        normalization = None
        K = None

    # use only enough cores as length of model list
    if args.numcores > len(model_names):
        args.numcores = len(model_names)


    # define the seed set for the iterations. Allows for defining in the configuration file
    try:
        seeds = config["seeds"]
    except:
        seeds = [0]
        print(f"Did not find 'seeds' in config file, defaulting to : {seeds}")
        
    rate = config.get("initial_labels_per_class", 1)
    

    # Iterations for the different tests
    for it, seed in enumerate(seeds):
        # For seed-dependent reps, we need labels before generating the trainset.
        # Load labels cheaply from base graph/data if they are not already available.
        if labels is None:
            _, labels_tmp, _, _, _ = load_graph(
                args.dataset,
                args.metric,
                numeigs=None,
                knn=args.knn,
                rep_cfg=None,
                returnK=True,
            )
            labels = labels_tmp

        # get initially labeled indices, based on the given seed
        labeled_ind = gl.trainsets.generate(labels, rate=rate, seed=seed)

        # For seed-dependent reps (nn), now build the graph/models using this seed's initial labels
        # Build graph/models for this seed.
        if representation_depends_on_seed(rep_cfg):
            if dynamic_rep:
                # For dynamic NN, each acquisition run rebuilds its own graph/model state.
                # We only need labels here to generate the initial labeled set.
                trainset = None
                normalization = None
                K = np.unique(labels).size
                models = [None] * len(model_names)
            else:
                models, labels, trainset, normalization, K = get_graph_and_models(
                    acq_funcs_names,
                    model_names,
                    args,
                    rep_cfg=rep_cfg,
                    labeled_ind=labeled_ind,
                    seed=seed,
                    rate=rate
                )

        # if manually pass in K value in command line then overwrite value of K
        K_current = args.K if args.K != 0 else K

        # define the results directory for this seed's test
        RESULTS_DIR = os.path.join(args.resultsdir,f"{args.dataset}_{rep_tag}_r{rate}_results_{seed}_{args.iters}")
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        np.save(os.path.join(RESULTS_DIR, "init_labeled.npy"), labeled_ind) # save initially labeled points that are common to each test


        def active_learning_test(acq_func_name, model_name, model):
            """
            Active learning test definition for parallelization.
            Handles both static and dynamic representation modes.
            """

            choices_run_savename = os.path.join(RESULTS_DIR, f"choices_{acq_func_name}_{model_name}.npy")
            acc_dir = os.path.join(RESULTS_DIR, model_name)
            acc_fname = os.path.join(acc_dir, f"acc_{acq_func_name}_{model_name}.npy")

            if os.path.exists(choices_run_savename) and os.path.exists(acc_fname):
                print(f"Found choices and acc for {acq_func_name} in {model_name}")
                return

            if not os.path.exists(acc_dir):
                os.makedirs(acc_dir)

            # STATIC PATH ====================
            if not dynamic_rep:
                if "decaytau" in acq_func_name:
                    eps = 1e-9
                    mu = (eps / model.tau)**(.5 / K_current)

                AL = get_active_learner(acq_func_name, model, labeled_ind, labels[labeled_ind], normalization, args)
                if "prop" in acq_func_name:
                    AL.acq_function.set_K(K_current)

                if trainset is None:
                    candidate_ind_all = np.arange(model.graph.num_nodes)
                else:
                    candidate_ind_all = trainset.copy()

                if acq_func_name[-3:] == 'kde':
                    knn_ind, knn_dist = gl.weightmatrix.load_knn_data(args.dataset.split("-")[0], metric=args.metric)
                    d = np.max(knn_dist, axis=1)
                    kde = (d / d.max())**(-1)
                    outlier_inds = np.where(kde < np.percentile(kde, 10))[0]
                    candidate_ind_all = np.setdiff1d(candidate_ind_all, outlier_inds)
                    print(f"Set candidate_ind for active learner of {acq_func_name} to throw out outliers")

                print(f"{acq_func_name}, training_set size = {candidate_ind_all.size}, dataset size = {model.graph.num_nodes}")

                acc = np.array([gl.ssl.ssl_accuracy(AL.model.predict(), labels, AL.labeled_ind)])

                for j in tqdm(range(args.iters), desc=f"{args.dataset}, {acq_func_name} test {it+1}/{len(seeds)}, seed = {seed}"):
                    query_points = AL.select_queries(candidate_ind=np.setdiff1d(candidate_ind_all, AL.labeled_ind))
                    query_labels = labels[query_points]
                    AL.update(query_points, query_labels)

                    if "decaytau" in acq_func_name:
                        if model.tau[0] != 0:
                            model.tau = mu * np.copy(model.tau)
                            if model.tau[0] < eps:
                                model.tau = np.zeros_like(model.tau)

                    acc = np.append(acc, gl.ssl.ssl_accuracy(AL.model.predict(), labels, AL.labeled_ind))

                np.save(acc_fname, acc)
                np.save(choices_run_savename, AL.labeled_ind)
                return

            # DYNAMIC PATH ====================
            labeled_seq = labeled_ind.copy()

            # track current tau for decaytau methods across graph rebuilds
            current_tau = None
            mu = None
            eps = 1e-9

            # initial build
            dyn_models, dyn_labels, dyn_trainset, dyn_normalization, dyn_K = build_dynamic_graph_and_models(
                acq_funcs_names,
                model_names,
                args,
                rep_cfg=rep_cfg,
                labeled_ind=labeled_seq,
            )
            dyn_model_dict = {name: mdl for name, mdl in zip(model_names, dyn_models)}
            current_model = dyn_model_dict[model_name]

            if "decaytau" in acq_func_name:
                current_tau = np.copy(current_model.tau)
                mu = (eps / current_tau)**(.5 / dyn_K)

            AL = get_active_learner(
                acq_func_name,
                current_model,
                labeled_seq,
                dyn_labels[labeled_seq],
                dyn_normalization,
                args,
            )
            if "prop" in acq_func_name:
                AL.acq_function.set_K(dyn_K)

            if dyn_trainset is None:
                candidate_ind_all = np.arange(current_model.graph.num_nodes)
            else:
                candidate_ind_all = dyn_trainset.copy()

            if acq_func_name[-3:] == 'kde':
                knn_ind, knn_dist = gl.weightmatrix.load_knn_data(args.dataset.split("-")[0], metric=args.metric)
                d = np.max(knn_dist, axis=1)
                kde = (d / d.max())**(-1)
                outlier_inds = np.where(kde < np.percentile(kde, 10))[0]
                candidate_ind_all = np.setdiff1d(candidate_ind_all, outlier_inds)
                print(f"Set candidate_ind for active learner of {acq_func_name} to throw out outliers")

            print(f"[DYNAMIC] {acq_func_name}, training_set size = {candidate_ind_all.size}, dataset size = {current_model.graph.num_nodes}")
            acc = np.array([gl.ssl.ssl_accuracy(AL.model.predict(), dyn_labels, AL.labeled_ind)])

            for j in tqdm(range(args.iters), desc=f"{args.dataset}, {acq_func_name} dynamic test {it+1}/{len(seeds)}, seed = {seed}"):
                query_points = AL.select_queries(candidate_ind=np.setdiff1d(candidate_ind_all, AL.labeled_ind))
                query_labels = dyn_labels[query_points]

                labeled_seq = np.concatenate([labeled_seq, np.atleast_1d(query_points)])

                # decay tau after acquiring the new point
                if "decaytau" in acq_func_name and current_tau is not None:
                    if current_tau[0] != 0:
                        current_tau = mu * np.copy(current_tau)
                        if current_tau[0] < eps:
                            current_tau = np.zeros_like(current_tau)

                rebuild_graph = (
                    nn_update_interval > 0 and
                    ((j + 1) % nn_update_interval == 0)
                )

                if rebuild_graph:
                    dyn_models, dyn_labels, dyn_trainset, dyn_normalization, dyn_K = build_dynamic_graph_and_models(
                        acq_funcs_names,
                        model_names,
                        args,
                        rep_cfg=rep_cfg,
                        labeled_ind=labeled_seq,
                    )
                    dyn_model_dict = {name: mdl for name, mdl in zip(model_names, dyn_models)}
                    current_model = dyn_model_dict[model_name]

                    if current_tau is not None:
                        current_model.tau = np.copy(current_tau)

                    AL = get_active_learner(
                        acq_func_name,
                        current_model,
                        labeled_seq,
                        dyn_labels[labeled_seq],
                        dyn_normalization,
                        args,
                    )
                    if "prop" in acq_func_name:
                        AL.acq_function.set_K(dyn_K)

                    if dyn_trainset is None:
                        candidate_ind_all = np.arange(current_model.graph.num_nodes)
                    else:
                        candidate_ind_all = dyn_trainset.copy()

                    if acq_func_name[-3:] == 'kde':
                        knn_ind, knn_dist = gl.weightmatrix.load_knn_data(args.dataset.split("-")[0], metric=args.metric)
                        d = np.max(knn_dist, axis=1)
                        kde = (d / d.max())**(-1)
                        outlier_inds = np.where(kde < np.percentile(kde, 10))[0]
                        candidate_ind_all = np.setdiff1d(candidate_ind_all, outlier_inds)
                else:
                    # reuse same graph, just rebuild model cleanly
                    current_model = type(AL.model)(AL.model.graph)

                    if current_tau is not None:
                        current_model.tau = np.copy(current_tau)

                    AL = get_active_learner(
                        acq_func_name,
                        current_model,
                        labeled_seq,
                        dyn_labels[labeled_seq],
                        dyn_normalization,
                        args,
                    )
                    if "prop" in acq_func_name:
                        AL.acq_function.set_K(dyn_K)

                acc = np.append(acc, gl.ssl.ssl_accuracy(AL.model.predict(), dyn_labels, AL.labeled_ind))

            np.save(acc_fname, acc)
            np.save(choices_run_savename, labeled_seq)
            return

        print("------Starting Active Learning Tests-------")

        Parallel(n_jobs=args.numcores)(delayed(active_learning_test)(acq_name, mdlname, mdl) for acq_name, mdlname, mdl \
                in zip(acq_funcs_names, model_names, models))
