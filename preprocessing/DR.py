# Imports
import umap
from torch_geometric.utils import to_undirected, add_self_loops
from scipy.spatial import Delaunay
from torch_geometric.nn import GCNConv
import torch.nn as nn
from torch.nn import ReLU
import torch
import torch_geometric
import networkx as nx

def create_delaunay_graph(positions):
    positions = positions.cpu().numpy()
    delaunay = Delaunay(positions,qhull_options='QJ')

    edges = []
    for simplex in delaunay.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edge = (simplex[i], simplex[j])
                edgess = (simplex[j], simplex[i])
                edges.append(edge)
                edges.append(edgess)

    delaunay_graph = nx.Graph(edges)

    edge_index = torch.tensor(list(delaunay_graph.edges)).t().contiguous()
    data = torch_geometric.data.Data(edge_index=edge_index)
    return data

class GCNWithEmbeddings(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.conv2 = GCNConv(hidden_dim, output_dim)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        self.embeddings = x.detach()  # Store embeddings after first layer
        x = self.conv2(x, edge_index)
        return x  # logits for training

def get_node_embeddings_from_gcn(data, is_undirected=False, device=None,
                                 hidden_dim=64, output_dim=16,
                                 lr=0.01, weight_decay=5e-4, max_epochs=200):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    data = data.to(device)

    # Ensure undirected
    edge_index = data.edge_index
    if is_undirected:
        edge_index = to_undirected(edge_index)

    # Prepare model and optimizer
    model = GCNWithEmbeddings(data.num_node_features, hidden_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    # Setup masks
    num_nodes = data.x.size(0)
    data.train_mask = torch.ones(num_nodes, dtype=torch.bool)

    # Train
    model.train()
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        out = model(data.x, edge_index)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    # Extract final learned embeddings
    embeddings = model.embeddings.cpu()
    return embeddings

def dr_method(data, is_undirected=False, device=None, save_dir='rewired_graphs', dataset_name=None, graph_index=0, debug=False):
    if (dataset_name == "cora" or dataset_name == "citeseer" or dataset_name == "pubmed" or dataset_name == "chameleon" or dataset_name == "actor" or dataset_name == "squirrel"):
        embeddings = get_node_embeddings_from_gcn(data, is_undirected=False, device=None,
                                 hidden_dim=64, output_dim=16,
                                 lr=0.01, weight_decay=5e-4, max_epochs=200)
        # Reduce graph features to 2D
        reducer = umap.UMAP(n_components=2)
        reduced_data = reducer.fit_transform(embeddings)

        # Construct DR graph
        new_data = create_delaunay_graph(torch.tensor(reduced_data))
        delauney_G = to_undirected(new_data.edge_index) # ensure undirected

        # Add self loops
        delauney_G,_= add_self_loops(delauney_G, num_nodes=data.num_nodes)
        
        # Get necessary outputs
        edge_index = delauney_G
        edge_type = torch.zeros(edge_index.size(1), dtype=torch.long)

        return edge_index, edge_type
    else:
        # Reduce graph features to 2D
        reducer = umap.UMAP(n_components=2)
        reduced_data = reducer.fit_transform(data.x)

        # Construct DR graph
        new_data = create_delaunay_graph(torch.tensor(reduced_data))
        delauney_G = to_undirected(new_data.edge_index) # ensure undirected

        # Add self loops
        delauney_G,_= add_self_loops(delauney_G, num_nodes=data.num_nodes)
        
        # Get necessary outputs
        edge_index = delauney_G
        edge_type = torch.zeros(edge_index.size(1), dtype=torch.long)

        return edge_index, edge_type

