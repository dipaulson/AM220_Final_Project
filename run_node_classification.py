from attrdict import AttrDict
from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor, Planetoid #LRGBDataset, HeterophilousGraphDataset
from torch_geometric.utils import to_networkx, from_networkx, to_undirected, dropout_adj
from torch_geometric.transforms import LargestConnectedComponents, ToUndirected
from experiments.node_classification import Experiment
from experiments.feature_shuffle import Experiment_shuffle

import time
import torch
import numpy as np
import pandas as pd
from hyperparams import get_args_from_input
from preprocessing import rewiring, sdrf, fosr, borf, DR

import torch_geometric.transforms as T
from torch_geometric.transforms import Compose


default_args = AttrDict({
    "dropout": 0.5, ## good
    "num_layers": 3, ## good
    "hidden_dim": 128, ## good
    "learning_rate": 1e-3, ## good
    "layer_type": "R-GCN", ## NEEDS TO BE SPECIFIED
    "display": True, ## good
    "num_trials": 100, ## good
    "eval_every": 1,
    "rewiring": "fosr", ## NEEDS TO BE SPECIFIED
    "num_iterations": 3, # HYPERPARAMETER
    "num_relations": 2,
    "patience": 100, # means 100 epochs
    "dataset": None,
    "borf_batch_add" : 4, # HYPERPARAMETER
    "borf_batch_remove" : 2, # HYPERPARAMETER
    "sdrf_remove_edges" : False,
    "encoding": None,
    "shuffle": False, # Added shuffle parameter, NEEDS TO BE SPECIFIED
    "ratio": 0.0 # Added shuffle ratio parameter, NEEDS TO BE SPECIFIED
})


results = []
args = default_args
args += get_args_from_input()

largest_cc = LargestConnectedComponents()

# Heterophilous
cornell = WebKB(root="data", name="Cornell", transform=largest_cc)
wisconsin = WebKB(root="data", name="Wisconsin", transform=largest_cc)
texas = WebKB(root="data", name="Texas", transform=largest_cc)
chameleon = WikipediaNetwork(root="data", name="chameleon", transform=largest_cc)
squirrel = WikipediaNetwork(root="data", name="squirrel", transform=largest_cc)
actor = Actor(root="data", transform=largest_cc)

# Homophilous
cora = Planetoid(root="data", name="cora", transform=largest_cc)
citeseer = Planetoid(root="data", name="citeseer", transform=largest_cc)
pubmed = Planetoid(root="data", name="pubmed", transform=largest_cc)

# All datasets
"""
datasets = {"cornell": cornell, "wisconsin": wisconsin, "texas": texas,
            "chameleon": chameleon, "squirrel": squirrel, "actor": actor,
            "cora": cora, "citeseer": citeseer, "pubmed": pubmed}
"""

# Dataset for testing
datasets = {"cora": cora}

for key in datasets:
    dataset = datasets[key]
    dataset.data.edge_index = to_undirected(dataset.data.edge_index)

def log_to_file(message, filename="results/node_classification.txt"):
    print(message)
    file = open(filename, "a")
    file.write(message)
    file.close()


if args.dataset:
    # restricts to just the given dataset if this mode is chosen
    name = args.dataset
    datasets = {name: datasets[name]}


for key in datasets:
    accuracies = []
    print(f"TESTING: {key} ({args.rewiring})")
    dataset = datasets[key]

    start = time.time()
    if args.rewiring == "fosr":
        edge_index, edge_type, _ = fosr.edge_rewire(dataset.data.edge_index.numpy(), num_iterations=args.num_iterations)
        dataset.data.edge_index = torch.tensor(edge_index)
        dataset.data.edge_type = torch.tensor(edge_type)
        print(dataset.data.num_edges)
        print(len(dataset.data.edge_type))
    elif args.rewiring == "sdrf_bfc":
        curvature_type = "bfc"
        dataset.data.edge_index, dataset.data.edge_type = sdrf.sdrf(dataset.data, loops=args.num_iterations, remove_edges=args.sdrf_remove_edges, 
                is_undirected=True, curvature=curvature_type)
    elif args.rewiring == "borf":
        print(f"[INFO] BORF hyper-parameter : num_iterations = {args.num_iterations}")
        print(f"[INFO] BORF hyper-parameter : batch_add = {args.borf_batch_add}")
        print(f"[INFO] BORF hyper-parameter : num_iterations = {args.borf_batch_remove}")
        dataset.data.edge_index, dataset.data.edge_type = borf.borf3(dataset.data, 
                loops=args.num_iterations, 
                remove_edges=False, 
                is_undirected=True,
                batch_add=args.borf_batch_add,
                batch_remove=args.borf_batch_remove,
                dataset_name=key,
                graph_index=0)
    elif args.rewiring == "AFR3":
        print(f"[INFO] AFR3 hyper-parameter : num_iterations = {args.num_iterations}")
        print(f"[INFO] AFR3 hyper-parameter : batch_add = {args.borf_batch_add}")
        print(f"[INFO] AFR3 hyper-parameter : batch_remove = {args.borf_batch_remove}")
        for i in range(len(dataset)):
            dataset[i].edge_index, dataset[i].edge_type = borf.borf4(dataset[i], 
                    loops=args.num_iterations, 
                    remove_edges=False, 
                    is_undirected=True,
                    batch_add=args.borf_batch_add,
                    batch_remove=args.borf_batch_remove,
                    dataset_name=key,
                    graph_index=i)
    elif args.rewiring == "AFR4":
        print(f"[INFO] AFR4 hyper-parameter : num_iterations = {args.num_iterations}")
        print(f"[INFO] AFR4 hyper-parameter : batch_add = {args.borf_batch_add}")
        print(f"[INFO] AFR4 hyper-parameter : batch_remove = {args.borf_batch_remove}")
        for i in range(len(dataset)):
            dataset[i].edge_index, dataset[i].edge_type = borf.borf5(dataset[i], 
                    loops=args.num_iterations, 
                    remove_edges=False, 
                    is_undirected=True,
                    batch_add=args.borf_batch_add,
                    batch_remove=args.borf_batch_remove,
                    dataset_name=key,
                    graph_index=i)
    elif args.rewiring == "sdrf_orc":
        curvature_type = "orc"
        dataset.data.edge_index, dataset.data.edge_type = sdrf.sdrf(dataset.data, loops=args.num_iterations, remove_edges=False, 
                is_undirected=True, curvature=curvature_type)
        
    elif args.rewiring == "dropedge":
        p = 0.8
        print(f"[INFO] Dropping edges with probability {p}")
        for i in range(len(dataset)):
            dataset[i].edge_index, dataset[i].edge_type = dropout_adj(dataset[i].edge_index, dataset[i].edge_type, p=p, force_undirected=True)

    # adding in baseline with no rewiring
    elif args.rewiring == "none":
        print(f"[INFO] No rewiring applied")

    elif args.rewiring == "DR":
        print(f"[INFO] DR rewiring")
        for i in range(len(dataset)):
            dataset[i].edge_index, dataset[i].edge_type = DR.dr_method(dataset[i], is_undirected=True, dataset_name=key, graph_index=i)

    end = time.time()
    rewiring_duration = end - start
    print(f"Rewiring duration: {rewiring_duration}")


    if (args.shuffle):
        # print(rewiring.spectral_gap(to_networkx(dataset.data, to_undirected=True)))
        start = time.time()
        for trial in range(args.num_trials):
            print(f"TRIAL #{trial+1}")
            test_accs = []
            for i in range(args.num_splits):
                train_acc, validation_acc, test_acc = Experiment_shuffle(args=args, dataset=dataset).run()
                test_accs.append(test_acc)
            test_acc = max(test_accs)
            accuracies.append(test_acc)
        end = time.time()
        run_duration = end - start 
    else:
        # print(rewiring.spectral_gap(to_networkx(dataset.data, to_undirected=True)))
        start = time.time()
        for trial in range(args.num_trials):
            print(f"TRIAL #{trial+1}")
            test_accs = []
            for i in range(args.num_splits):
                train_acc, validation_acc, test_acc = Experiment(args=args, dataset=dataset).run()
                test_accs.append(test_acc)
            test_acc = max(test_accs)
            accuracies.append(test_acc)
        end = time.time()
        run_duration = end - start

    log_to_file(f"RESULTS FOR {key} ({args.rewiring}):\n")
    log_to_file(f"average acc: {np.mean(accuracies)}\n")
    log_to_file(f"plus/minus:  {2 * np.std(accuracies)/(args.num_trials ** 0.5)}\n\n")
    results.append({
        "dataset": key,
        "rewiring": args.rewiring,
        "num_iterations": args.num_iterations,
        "borf_batch_add" : args.borf_batch_add,
        "borf_batch_remove" : args.borf_batch_remove,
        "avg_accuracy": np.mean(accuracies),
        "ci":  2 * np.std(accuracies)/(args.num_trials ** 0.5),
        "run_duration" : run_duration,
        #"rewiring_duration" : rewiring_duration
    })
    results_df = pd.DataFrame(results)
    with open(f'results/node_classification_{args.layer_type}_{args.rewiring}.csv', 'a') as f:
        results_df.to_csv(f, mode='a', header=f.tell()==0)