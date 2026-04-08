import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import os

# 3sigma
def get_threshold(k, after_combine_matrix):
    sigma = np.std(after_combine_matrix, axis=1, ddof=1)
    mean = np.mean(after_combine_matrix, axis=1)
    F = 1 / (1 + sigma ** 2)
    th = mean + (k * sigma)*(1-F)
    return th

def construct_of_1st_layer(data_file, nodes_names, SPIN, label, species, dataset, k):
    genedata = pd.read_csv(data_file)
    gene_expression_name = np.array(genedata.iloc[:, 0])
    gene_expression_value = np.array(genedata.iloc[:, 1:])
    nodes_expression = []
    for name in nodes_names:
        matched = False
        for sub_name in str(name).split('|'):
            sub_name = sub_name.strip()
            if sub_name in gene_expression_name:
                ind = np.argwhere(gene_expression_name == sub_name)
                nodes_expression.append(gene_expression_value[ind[0, 0], :])
                matched = True
                break
        if not matched:
            nodes_expression.append(np.zeros(gene_expression_value.shape[1]))
    nodes_expression = np.array(nodes_expression)
    print("gene expression profile",nodes_expression.shape)
    fea_gene_save_path = f"../datasets/%s/PPI/%s/refined_net_and_fea/fea_gene.npy" % (species, dataset)
    if not os.path.exists(fea_gene_save_path):
        os.makedirs(os.path.dirname(fea_gene_save_path), exist_ok=True)
        np.save(fea_gene_save_path, nodes_expression)
        print(f"Finished fea_gene.npy!")
    else:
        print(f"fea_gene.npy already exists, skipped saving!")

    threshold = get_threshold(k, nodes_expression)
    print(len(threshold))
    nodes_expression_0_1 = np.zeros_like(nodes_expression)
    for i in range(len(nodes_names)):
        for j in range(gene_expression_value.shape[1]):
            if threshold[i] < nodes_expression[i, j]:
                nodes_expression_0_1[i, j] = 1
    fea_act_save_path = f"../datasets/%s/PPI/%s/refined_net_and_fea/fea_act.npy" % (species, dataset)
    if not os.path.exists(fea_act_save_path):
        os.makedirs(os.path.dirname(fea_act_save_path), exist_ok=True)
        np.save(fea_act_save_path, nodes_expression_0_1)
        print(f"Finished fea_act.npy!")
    else:
        print(f"fea_act.npy already exists, skipped saving!")

    dynamic_network = []
    for k in range(nodes_expression_0_1.shape[1]):
        print("k= ", k)
        DPIN = np.zeros((len(nodes_names), len(nodes_names)))
        for i in range(len(nodes_names)):
            for j in range(i + 1, len(nodes_names)):
                if nodes_expression_0_1[i, k] == nodes_expression_0_1[j, k] == 1:
                    DPIN[i, j] = DPIN[j, i] = 1
        DPIN = DPIN * SPIN

        source_index_1st = []
        target_index_1st = []
        edge_weight = []
        for i in range(nodes_expression_0_1.shape[0]):
            source_index_1st.append(i)
            target_index_1st.append(i)
            edge_weight.append(1)
            for j in range(i + 1, nodes_expression_0_1.shape[0]):
                if DPIN[i, j] != 0:
                    source_index_1st.append(i)
                    target_index_1st.append(j)
                    edge_weight.append(DPIN[i, j])
                    source_index_1st.append(j)
                    target_index_1st.append(i)
                    edge_weight.append(DPIN[i, j])
        edge_index = torch.tensor([source_index_1st, target_index_1st], dtype=torch.int64)
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)
        sub_dynamic_network = Data(x=nodes_expression[:, :], y=label, edge_index=edge_index, edge_weight=edge_weight,
                                   num_classes=2)
        dynamic_network.append(sub_dynamic_network)
    print(dynamic_network)
    return dynamic_network

def dy_net_construction(species, dataset, k_value):

    nodes_names = np.load(r"../datasets/%s/PPI/%s/gene_name.npy"%(species,dataset), allow_pickle=True)
    SPIN = np.load(r'../datasets/%s/PPI/%s/SPIN.npy'%(species,dataset), allow_pickle=True)
    label = np.load(r'../datasets/%s/PPI/%s/labels.npy'%(species,dataset))
    print("The number of interactions in the SPIN: ", sum(sum(SPIN))/2)

    gene_data_path = "../datasets/%s/Expression/profile.csv" % (species)
    dy_net_save_path = f"../datasets/%s/PPI/%s/refined_net_and_fea/dynamic_network_k=%.2f_all_fea_(1-F).pt" % (
        species, dataset, k_value)
    if not os.path.exists(dy_net_save_path):
        os.makedirs(os.path.dirname(dy_net_save_path), exist_ok=True)
        dyn = construct_of_1st_layer(gene_data_path, nodes_names, SPIN, label, species, dataset, k_value)
        torch.save(dyn, dy_net_save_path)
        print(f"Finished dyn.torch!")
    else:
        print(f"dyn.torch already exists, skipped saving!")
