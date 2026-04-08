import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SpatialGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim1, out_dim2, drop_conv):
        super().__init__()
        self.gcn1 = GCNConv(in_dim, out_dim1, add_self_loops=False)
        self.bn1 = nn.BatchNorm1d(out_dim1, track_running_stats=False)
        self.dropout1 = nn.Dropout(drop_conv)

        self.gcn2 = GCNConv(out_dim1, out_dim2, add_self_loops=False)
        self.bn2 = nn.BatchNorm1d(out_dim2, track_running_stats=False)
        self.dropout2 = nn.Dropout(drop_conv)

    def forward(self, x, edge_index, edge_weight):
        x = self.gcn1(x, edge_index, edge_weight=edge_weight)
        x = self.bn1(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        x = self.dropout1(F.relu(x))

        x = self.gcn2(x, edge_index, edge_weight=edge_weight)
        x = self.bn2(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        x = self.dropout2(F.relu(x))
        return x


class TemporalSelfAttention(nn.Module):
    def __init__(self, hidden_dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.last_attn = None

    def forward(self, x_seq):
        out, attn = self.attn(
            x_seq, x_seq, x_seq,
            need_weights=True,
            average_attn_weights=True
        )
        self.last_attn = attn
        out = self.norm(out)
        out = out.mean(dim=1)
        return out


class Dyn_emb(nn.Module):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2, drop_conv):
        super().__init__()
        self.spatial = SpatialGCNLayer(in_dim, hidden_dim1, hidden_dim2, drop_conv)

    def forward(self, snapshots, device):
        z_seq = []
        for data in snapshots:
            x = data.x
            edge_index = data.edge_index
            edge_weight = data.edge_weight

            if isinstance(x, torch.Tensor) and x.ndim == 1:
                x = x.unsqueeze(-1)
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float)

            x = x.to(device)
            edge_index = edge_index.to(device)
            edge_weight = edge_weight.to(device)

            z_t = self.spatial(x, edge_index, edge_weight)
            z_seq.append(z_t.unsqueeze(1))

        z_all = torch.cat(z_seq, dim=1)
        return z_all


def fast_graph_similarity(edge_index1, edge_index2):
    e1 = set(map(tuple, edge_index1.t().tolist()))
    e2 = set(map(tuple, edge_index2.t().tolist()))
    inter = len(e1 & e2)
    union = len(e1 | e2)
    return inter / union if union > 0 else 0.0


class DyHTSAGCN(nn.Module):
    def __init__(self, args):
        super(DyHTSAGCN, self).__init__()

        self.species = args.species
        self.dataset = args.dataset
        self.k_value = args.k_value

        self.gene_layer_input = args.gene_layer_input
        self.sub_layer_input = args.sub_layer_input
        self.orth_layer_input = args.orth_layer_input

        self.conv_hidden1 = args.conv_hidden1
        self.conv_hidden2 = args.conv_hidden2
        self.drop_mlp = args.drop_mlp
        self.drop_conv = args.drop_conv
        self.mlp_hidden = args.mlp_hidden
        self.classifier_hidden_1 = args.classifier_hidden_1
        self.classifier_hidden_2 = args.classifier_hidden_2
        self.drop_classifier = args.drop_classifier
        self.T = args.T

        self.dyn_emb = Dyn_emb(
            in_dim=self.gene_layer_input * 2,
            hidden_dim1=self.conv_hidden1,
            hidden_dim2=self.conv_hidden2,
            drop_conv=self.drop_conv
        )
        self.temporal_attention = TemporalSelfAttention(hidden_dim=self.conv_hidden2)
        self.bn_dyn = nn.BatchNorm1d(self.conv_hidden2, track_running_stats=False)

        self.dropout_mlp = nn.Dropout(self.drop_mlp)
        self.dropout_cov = nn.Dropout(self.drop_conv)
        self.classifier_drop = nn.Dropout(self.drop_classifier)

        self.conv_4th_1 = GCNConv(self.orth_layer_input, self.conv_hidden1, add_self_loops=False)
        self.conv_4th_2 = GCNConv(self.conv_hidden1, self.conv_hidden2, add_self_loops=False)
        self.bn_4th_1 = nn.BatchNorm1d(self.conv_hidden1, track_running_stats=False)
        self.bn_4th_2 = nn.BatchNorm1d(self.conv_hidden2, track_running_stats=False)

        self.bn_mlp_1 = nn.BatchNorm1d(self.mlp_hidden, track_running_stats=False)
        self.bn_mlp_2 = nn.BatchNorm1d(self.conv_hidden2, track_running_stats=False)
        self.bn_cla_1 = nn.BatchNorm1d(self.classifier_hidden_1, track_running_stats=False)
        self.bn_cla_2 = nn.BatchNorm1d(self.classifier_hidden_2, track_running_stats=False)

        if self.species != "coli":
            self.conv_2nd_1 = GCNConv(self.sub_layer_input, self.conv_hidden1, add_self_loops=False)
            self.conv_2nd_2 = GCNConv(self.conv_hidden1, self.conv_hidden2, add_self_loops=False)
            self.bn_2nd_1 = nn.BatchNorm1d(self.conv_hidden1, track_running_stats=False)
            self.bn_2nd_2 = nn.BatchNorm1d(self.conv_hidden2, track_running_stats=False)

            self.mlp_for_fea = nn.Sequential(
                nn.Linear(
                    self.gene_layer_input + self.sub_layer_input + self.orth_layer_input,
                    self.mlp_hidden
                ),
                self.bn_mlp_1,
                nn.ReLU(),
                self.dropout_mlp,
                nn.Linear(self.mlp_hidden, self.conv_hidden2),
                self.bn_mlp_2,
                nn.ReLU(),
                self.dropout_mlp
            )

            self.attention_layer = nn.MultiheadAttention(
                embed_dim=4 * self.conv_hidden2,
                num_heads=8,
                batch_first=True
            )

            self.classifier_1 = nn.Sequential(
                nn.Linear(4 * self.conv_hidden2, self.classifier_hidden_1),
                self.bn_cla_1,
                nn.ReLU(),
                self.classifier_drop,
                nn.Linear(self.classifier_hidden_1, self.classifier_hidden_2),
                self.bn_cla_2,
                nn.ReLU(),
                self.classifier_drop,
                nn.Linear(self.classifier_hidden_2, 1)
            )
        else:
            self.mlp_for_fea = nn.Sequential(
                nn.Linear(self.gene_layer_input + self.orth_layer_input, self.mlp_hidden),
                self.bn_mlp_1,
                nn.ReLU(),
                self.dropout_mlp,
                nn.Linear(self.mlp_hidden, self.conv_hidden2),
                self.bn_mlp_2,
                nn.ReLU(),
                self.dropout_mlp
            )

            self.attention_layer = nn.MultiheadAttention(
                embed_dim=3 * self.conv_hidden2,
                num_heads=4,
                batch_first=True
            )

            self.classifier_1 = nn.Sequential(
                nn.Linear(3 * self.conv_hidden2, self.classifier_hidden_1),
                self.bn_cla_1,
                nn.ReLU(),
                self.classifier_drop,
                nn.Linear(self.classifier_hidden_1, self.classifier_hidden_2),
                self.bn_cla_2,
                nn.ReLU(),
                self.classifier_drop,
                nn.Linear(self.classifier_hidden_2, 1)
            )

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

        mapped_path = (
            f"datasets/{self.species}/PPI/{self.dataset}/refined_net_and_fea/"
            f"dynamic_network_k={self.k_value:.2f}_all_fea_(1-F).pt"
        )
        active_path = (
            f"datasets/{self.species}/PPI/{self.dataset}/refined_net_and_fea/fea_act.npy"
        )

        mapped_graphs = torch.load(mapped_path, weights_only=False)
        self.x_active_all = torch.tensor(np.load(active_path), dtype=torch.float)

        for i in range(len(mapped_graphs)):
            x_gene_ori = mapped_graphs[i].x
            if not isinstance(x_gene_ori, torch.Tensor):
                x_gene_ori = torch.tensor(x_gene_ori, dtype=torch.float)
            x_combined = torch.cat([x_gene_ori, self.x_active_all], dim=1)
            mapped_graphs[i].x = x_combined

        self.snapshots_all = mapped_graphs
        self.sim_weight_cache = {}
        self.node_stability_cache = {}
        self._build_cache()

    def _build_cache(self):
        num_snapshots = len(self.snapshots_all)
        n_nodes = self.snapshots_all[0].num_nodes

        for t in range(num_snapshots):
            start_idx = max(0, t - self.T + 1)
            input_snapshots = self.snapshots_all[start_idx:t + 1]

            for k in range(len(input_snapshots)):
                self.sim_weight_cache[(t, k)] = fast_graph_similarity(
                    self.snapshots_all[t].edge_index,
                    input_snapshots[k].edge_index
                )

            act_matrix = self.x_active_all[:, start_idx:t + 1].T
            edge_indices = [snap.edge_index for snap in input_snapshots]
            stability_tensor = self.node_importance_score_with_activation(edge_indices, act_matrix)

            for i in range(n_nodes):
                self.node_stability_cache[(t, i)] = stability_tensor[i].item()

    def node_importance_score_with_activation(self, edge_indices, act_matrix):
        t_size, n_nodes = act_matrix.shape
        degrees = torch.zeros((t_size, n_nodes))

        for t, edge_index in enumerate(edge_indices):
            deg_t = torch.zeros(n_nodes)
            deg_t.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1)))
            degrees[t] = deg_t

        var_deg = torch.var(degrees, dim=0, unbiased=False)
        var_act = torch.var(act_matrix.float(), dim=0, unbiased=False)
        presence_d = (degrees > 0).float().mean(dim=0)
        presence_a = (act_matrix > 0).float().mean(dim=0)

        var_deg = torch.nan_to_num(var_deg, nan=0.0)
        var_act = torch.nan_to_num(var_act, nan=0.0)

        score = torch.exp(
            -self.alpha * var_deg / (presence_d + 1e-8)
            -self.beta * var_act / (presence_a + 1e-8)
        )
        return score

    def extract_embeddings(self, data):
        device = next(self.parameters()).device
        dyn_outputs = []
        num_snapshots = len(self.snapshots_all)
        n_nodes = self.snapshots_all[0].num_nodes

        for t in range(num_snapshots):
            start_idx = max(0, t - self.T + 1)
            input_snapshots = self.snapshots_all[start_idx:t + 1]

            out_dysat_seq = self.dyn_emb(input_snapshots, device=device)

            sim_weights = torch.tensor(
                [self.sim_weight_cache[(t, k)] for k in range(len(input_snapshots))],
                dtype=torch.float
            ).to(device).softmax(dim=0).view(1, -1, 1)

            out_dysat_t = (out_dysat_seq * sim_weights).sum(dim=1)

            node_weights = torch.tensor(
                [self.node_stability_cache[(t, i)] for i in range(n_nodes)],
                dtype=torch.float
            ).to(device).view(-1, 1)

            out_dysat_t = out_dysat_t * node_weights

            if torch.isnan(out_dysat_t).any():
                print(f"NaN detected at time {t} in node embeddings")
                out_dysat_t = torch.nan_to_num(out_dysat_t, nan=0.0, posinf=1e4, neginf=-1e4)

            dyn_outputs.append(out_dysat_t.unsqueeze(1))

        dysat_outputs = torch.cat(dyn_outputs, dim=1)
        out_dysat = self.temporal_attention(dysat_outputs)
        out_dysat = self.bn_dyn(out_dysat)
        out_dysat = F.relu(out_dysat)
        out_dysat = self.dropout_cov(out_dysat)

        x = data[0].x

        if self.species != "coli":
            edge_index2 = data[0].edge_index
            edge_index4 = data[1].edge_index

            out2 = self.conv_2nd_1(
                x[:, self.gene_layer_input:self.gene_layer_input + self.sub_layer_input],
                edge_index2
            )
            out2 = self.bn_2nd_1(out2)
            out2 = self.dropout_cov(F.relu(out2))
            out2 = self.conv_2nd_2(out2, edge_index2)
            out2 = self.bn_2nd_2(out2)
            out2 = self.dropout_cov(F.relu(out2))

            out4 = self.conv_4th_1(
                x[:, self.gene_layer_input + self.sub_layer_input:],
                edge_index4
            )
            out4 = self.bn_4th_1(out4)
            out4 = self.dropout_cov(F.relu(out4))
            out4 = self.conv_4th_2(out4, edge_index4)
            out4 = self.bn_4th_2(out4)
            out4 = self.dropout_cov(F.relu(out4))
        else:
            edge_index4 = data[0].edge_index
            out4 = self.conv_4th_1(x[:, self.gene_layer_input:], edge_index4)
            out4 = self.bn_4th_1(out4)
            out4 = self.dropout_cov(F.relu(out4))
            out4 = self.conv_4th_2(out4, edge_index4)
            out4 = self.bn_4th_2(out4)
            out4 = self.dropout_cov(F.relu(out4))

        fea_concat = self.mlp_for_fea(x)

        if self.species != "coli":
            out_all_mrpin = torch.cat((out_dysat, out2, out4, fea_concat), dim=1)
        else:
            out_all_mrpin = torch.cat((out_dysat, out4, fea_concat), dim=1)

        out_all_mrpin_expanded = out_all_mrpin.unsqueeze(1)
        attn_output, _ = self.attention_layer(
            out_all_mrpin_expanded,
            out_all_mrpin_expanded,
            out_all_mrpin_expanded
        )
        attn_output = attn_output.squeeze(1)
        return attn_output

    def forward(self, data):
        out = self.extract_embeddings(data)
        return self.classifier_1(out).squeeze(-1)