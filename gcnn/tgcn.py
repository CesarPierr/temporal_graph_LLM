# -*- coding: utf-8 -*-

import os
import numpy as np
from tqdm import tqdm
import torch
import pickle

import warnings
warnings.filterwarnings('ignore')

from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tgb.linkproppred.evaluate import Evaluator

DATA = "tkgl-smallpedia"

# data loading
dataset = PyGLinkPropPredDataset(name=DATA, root="datasets")
train_mask = dataset.train_mask
val_mask = dataset.val_mask
test_mask = dataset.test_mask
data = dataset.get_TemporalData()
metric = dataset.eval_metric

print ("there are {} nodes and {} edges".format(dataset.num_nodes, dataset.num_edges))
print ("there are {} relation types".format(dataset.num_rels))

dataset.load_val_ns()

evaluator = Evaluator(name=DATA)

train_years = data[train_mask].t
val_years = data[val_mask].t
test_years = data[test_mask].t

# Get sorted unique years
train_years = np.sort(np.unique(train_years))
val_years = np.sort(np.unique(val_years))
test_years = np.sort(np.unique(test_years))

if not os.path.exists('data.pkl'):
    formatted_data = {}

    for i in tqdm(range(len(data)), disable=True):
        src = data[i]["src"].item()
        dst = data[i]["dst"].item()
        t = data[i]["t"].item()
        edge_type = data[i]["edge_type"].item()
        if t not in formatted_data:
            formatted_data[t] = {'edge_index': [[], []], 'edge_type': []}
        formatted_data[t]['edge_index'][0].append(src)
        formatted_data[t]['edge_index'][1].append(dst)
        formatted_data[t]['edge_type'].append(edge_type)

    with open('data.pkl', 'wb') as f:
        pickle.dump(formatted_data, f)
else:
    with open('data.pkl', 'rb') as file:
        formatted_data = pickle.load(file)

for t in formatted_data:
    formatted_data[t]['edge_index'] = torch.tensor(formatted_data[t]['edge_index'])
    formatted_data[t]['edge_type'] = torch.tensor(formatted_data[t]['edge_type'])

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as g_nn

class TypeGCNConv(g_nn.MessagePassing):
    def __init__(self, in_channels, out_channels, num_rels):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_rels = num_rels

        self.weight = nn.Embedding(num_rels, in_channels * out_channels)


    def forward(self, x, edge_index, edge_type):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_type has shape [E]
        # E is the number of edges
        src = edge_index[0]
        degrees = (src.unsqueeze(0) == src.unsqueeze(1)).sum(axis=1) + 1

        edge_weight = self.weight(edge_type)
        edge_weight = edge_weight / (torch.norm(edge_weight, dim=1, keepdim=True)+1e-2)
        edge_weight = edge_weight / degrees.unsqueeze(1)
        edge_weight = edge_weight.view(-1, self.in_channels, self.out_channels)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        # x_j has shape [E, in_channels]
        # edge_weight has shape [E, in_channels, out_channels]
        return torch.einsum('ij,ijk->ik', x_j, edge_weight)

class TwoWayGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_rels):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        self.forward_conv = TypeGCNConv(in_channels, out_channels, num_rels)
        self.backward_conv = TypeGCNConv(in_channels, out_channels, num_rels)

    def forward(self, x, edge_index, edge_type):
        forward_edge_index = edge_index
        backward_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        forward_x = self.forward_conv(x, forward_edge_index, edge_type)
        backward_x = self.backward_conv(x, backward_edge_index, edge_type)
        x = self.lin(x) + forward_x + backward_x
        return x


class RecurrentGCN(nn.Module):
    def __init__(self, num_nodes, hidden_channels, out_channels, num_rels, num_layers=2, dropout=0.5):
        super(RecurrentGCN, self).__init__()

        self.in_layer = nn.Embedding(num_nodes, hidden_channels)
        self.rec_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            self.rec_layers.append(
                TwoWayGCNConv(hidden_channels, hidden_channels, num_rels)
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        self.out_layer = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, edge_index, edge_type):
        # x has shape [N]
        # edge_index has shape T x [2, E_t]
        # edge_type has shape T x [E_t]
        x = self.in_layer(x)
        T = len(edge_index)
        # print(x.min(), x.max())
        for t in range(T):
            for layer, batch_norm in zip(self.rec_layers, self.batch_norms):
                x = batch_norm(x)
                x = layer(x, edge_index[t], edge_type[t])
                x = F.relu(x)
                x = self.dropout(x)
                # print(x.min(), x.max())
        x = self.out_layer(x)
        # print(x.min(), x.max(), '\n')
        return x

class TemporalLinkPredictor(nn.Module):
    def __init__(self, num_nodes, hidden_channels, out_channels, num_rels, num_layers=2, dropout=0.5):
        super(TemporalLinkPredictor, self).__init__()

        self.num_nodes = num_nodes
        self.out_channels = out_channels

        self.gcn = RecurrentGCN(num_nodes, hidden_channels, out_channels, num_rels, num_layers, dropout)
        self.rel_embed = nn.Embedding(num_rels, out_channels * out_channels)
        self.src_proj = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.tgt_proj = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, src, tgt, edge_type, past_edge_data=None, x=None):
        assert x is not None or past_edge_data is not None, "Either x or past_edge_data must be provided"
        if x is None:
            x = torch.arange(self.num_nodes, device=src.device) # [N]
            x = self.gcn(x, **past_edge_data) # [N, out_channels]
        # print(x.shape, x.min(), x.max())
        # Get src embedding from graph embedding x
        src_embed = self.src_proj(x[src]) # [B, out_channels]
        # Get tgt embedding from graph embedding x
        tgt_embed = self.tgt_proj(x[tgt]) # [B, out_channels]
        edge_type_embed = self.rel_embed(edge_type).view(-1, out_channels, out_channels) # [B, out_channels, out_channels]
        # Compute scores: for each element in the batch, compute the score of the edge as src_embed @ edge_type_embed @ tgt_embed.T
        scores = torch.einsum("bij,bi->bj", edge_type_embed, tgt_embed) / (self.out_channels * self.out_channels)
        scores = (scores * src_embed).sum(axis=1)
        scores = F.sigmoid(scores)
        # print(scores.shape, scores.min(), scores.max())

        return scores

lr = 1e-5
weight_decay = 1e-6
hidden_channels = 64
out_channels = 64

num_epochs = 100
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TemporalLinkPredictor(dataset.num_nodes, hidden_channels, out_channels, dataset.num_rels, num_layers=2, dropout=0.5)
model = model.to(device)

# if os.path.exists('best_model.pt'):
#     state_dict = torch.load('best_model.pt')
#     model.load_state_dict(state_dict)
#     print("Loaded best_model.pt")
# elif os.path.exists('last_model.pt'):
if os.path.exists('last_model.pt'):
    state_dict = torch.load('last_model.pt')
    model.load_state_dict(state_dict)
    print("Loaded last_model.pt")
else:
    print("Initializing fresh model")

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

t_min = min(formatted_data.keys())
t_max = max(formatted_data.keys())

def train(model, optimizer, data, t, batch_size, window_size=None):
    model.train()
    optimizer.zero_grad()

    if window_size is None:
        tt_min = t_min
    else:
        tt_min = max(t_min, t-window_size)

    edge_data = data[t]
    # Get existing start nodes at t
    pos_src = edge_data['edge_index'][0, :]
    # Get existing end nodes at t
    pos_dst = edge_data['edge_index'][1, :]
    # Get existing edge types at t
    pos_edge_type = edge_data['edge_type']

    # Get negative samples
    neg_src = torch.randint(0, dataset.num_nodes, (len(pos_src),))
    neg_dst = torch.randint(0, dataset.num_nodes, (len(pos_src),))
    neg_edge_type = torch.randint(0, dataset.num_rels, (len(pos_src),))
    # Filter out negative samples that are positive
    mask = (neg_src != pos_src) | (neg_dst != pos_dst) | (neg_edge_type != pos_edge_type)
    neg_src = neg_src[mask]
    neg_dst = neg_dst[mask]
    neg_edge_type = neg_edge_type[mask]
    # Make sure we have the same number of negative samples as positive samples (we could have less if we filtered out some)
    neg_src = torch.cat([neg_src for _ in range(len(pos_src) // len(neg_src)+1)])[:len(pos_src)]
    neg_dst = torch.cat([neg_dst for _ in range(len(pos_src) // len(neg_dst)+1)])[:len(pos_src)]
    neg_edge_type = torch.cat([neg_edge_type for _ in range(len(pos_src) // len(neg_edge_type)+1)])[:len(pos_src)]

    # print(len(pos_src), len(pos_dst), len(pos_edge_type), len(neg_src), len(neg_dst), len(neg_edge_type))

    past_edge_data = {'edge_index': [data[tt]['edge_index'].to(device) for tt in range(tt_min, t)], 'edge_type': [data[tt]['edge_type'].to(device) for tt in range(tt_min, t)]}

    total_loss = 0

    train_indices = np.arange(len(pos_src))
    np.random.shuffle(train_indices)

    # Split into batches
    for i in range(0, len(pos_src), batch_size):
        batch_indices = train_indices[i:i+batch_size]
        batch_pos_src = pos_src[batch_indices].to(device)
        batch_pos_dst = pos_dst[batch_indices].to(device)
        batch_pos_edge_type = pos_edge_type[batch_indices].to(device)
        batch_neg_src = neg_src[batch_indices].to(device)
        batch_neg_dst = neg_dst[batch_indices].to(device)
        batch_neg_edge_type = neg_edge_type[batch_indices].to(device)

        batch_src = torch.cat([batch_pos_src, batch_neg_src])
        batch_dst = torch.cat([batch_pos_dst, batch_neg_dst])
        batch_edge_type = torch.cat([batch_pos_edge_type, batch_neg_edge_type])
        scores = model(batch_src, batch_dst, batch_edge_type, past_edge_data=past_edge_data, x=None)
        pos_scores = scores[:len(batch_pos_src)]
        neg_scores = scores[len(batch_pos_src):]

        eps = 1e-7
        pos_scores = torch.clamp(pos_scores, eps, 1-eps)
        neg_scores = torch.clamp(neg_scores, eps, 1-eps)

        pos_loss = -torch.log(pos_scores)
        neg_loss = -torch.log1p(-neg_scores)

        loss = pos_loss.mean() + neg_loss.mean()

        # print(t, i, torch.any(torch.isnan(pos_loss)), len(pos_loss), torch.any(torch.isnan(neg_loss)), len(neg_loss), loss.item())

        for n, p in model.named_parameters():
            if p.requires_grad:
                if torch.any(torch.isnan(p)):
                    raise ValueError(f'NaN in {n}')

        optimizer.zero_grad()
        loss.backward()

        for n, p in model.named_parameters():
            if p.requires_grad:
                if p.grad is None:
                    raise ValueError(f"{n} grad is None")
                if torch.any(torch.isnan(p.grad)):
                    raise ValueError(f'NaN in {n} grad')

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / (len(pos_src) // batch_size)

def validate(model, evaluator, data, dataset, t, batch_size, window_size=None):
    model.eval()

    if window_size is None:
        tt_min = t_min
    else:
        tt_min = max(t_min, t-window_size)

    edge_data = data[t]
    # Get existing start nodes at t
    pos_src = edge_data['edge_index'][0, :]
    # Get existing end nodes at t
    pos_dst = edge_data['edge_index'][1, :]
    # Get existing edge types at t
    pos_edge_type = edge_data['edge_type']
    # Duplicate t
    pos_timestamp = torch.tensor([t for _ in range(len(pos_src))])

    # Get negative samples
    neg_samples = dataset.dataset.ns_sampler.query_batch(pos_src, pos_dst, pos_timestamp, pos_edge_type, split_mode='val')

    past_edge_data = {'edge_index': [data[tt]['edge_index'].to(device) for tt in range(tt_min, t)], 'edge_type': [data[tt]['edge_type'].to(device) for tt in range(tt_min, t)]}

    # val_indices = np.arange(len(pos_src))

    total_metrics = {}
    val_indices = np.arange(len(pos_src))
    np.random.shuffle(val_indices)
    val_indices = val_indices[:batch_size]

    # Split into batches
    for i in val_indices:
        neg_dst = torch.tensor(neg_samples[i])
        N_neg = neg_dst.shape[0]
        
        batch_src = torch.repeat_interleave(pos_src[i], N_neg+1).to(device)
        batch_dst = torch.cat([pos_dst[i].view((1,)), neg_dst]).to(device)
        batch_edge_type = torch.repeat_interleave(pos_edge_type[i], N_neg+1).to(device)

        with torch.no_grad():
            scores = model(batch_src, batch_dst, batch_edge_type, past_edge_data=past_edge_data, x=None)
        pos_score = scores[0]
        neg_scores = scores[1:]

        input_dict = {"y_pred_pos": pos_score, "y_pred_neg": neg_scores, "eval_metric": ["mrr", "hits@"]}

        metrics = evaluator.eval(input_dict)
        for k, v in metrics.items():
            if not k in total_metrics:
                total_metrics[k] = v
            else:
                total_metrics[k] += v

    for k, v in total_metrics.items():
        total_metrics[k] = v / batch_size
    return  total_metrics

def train_loop():
    model.train()
    optimizer.zero_grad()
    train_count = 0
    total_loss = 0
    np.random.shuffle(indices)
    for t in train_years[indices]:
        if not t>=max(train_years):
            if t<=t_min:
                continue
            if np.random.random() < 0.8:
                continue
        train_count += 1
        loss = train(model, optimizer, formatted_data, t, batch_size, window_size=25)
        print(f"\tEpoch {epoch}, t={t}, loss={loss}")
        total_loss += loss
        torch.save(model.state_dict(), 'last_model.pt')
    total_loss /= train_count
    print(f"Epoch {epoch}: Total loss {total_loss}")


def val_loop():
    val_count = 0
    total_scores = {}
    for t in val_years:
        if not t>=max(val_years):
            if t<=t_min:
                continue
            if np.random.random() < 0.8:
                continue
        val_count += 1
        metrics = validate(model, evaluator, formatted_data, dataset, t, batch_size, window_size=25)
        scoring_str = ""
        for k, v in metrics.items():
            scoring_str += f"{k}: {v} "
            total_scores[k] = total_scores.get(k, 0) + v
        print(f"\tEpoch {epoch}, t={t}, {scoring_str}")
    total_scoring_str = ""
    for k, v in total_scores.items():
        total_scores[k] = v / val_count
        total_scoring_str += f"Total {k}: {v / val_count} "

    print(f"Epoch {epoch}: {total_scoring_str}")

    total_mrr = total_scores["mrr"]

    global best_mrr
    if total_mrr > best_mrr:
        best_mrr = total_mrr
        torch.save(model.state_dict(), 'best_model.pt')

epoch = -1
best_mrr = 0
# val_loop()

indices = np.arange(len(train_years))
# indices = np.arange(len(train_years))[-5:]
for epoch in range(num_epochs):
    train_loop()
    
    val_loop()

torch.save(model.state_dict(), 'last_model.pt')

