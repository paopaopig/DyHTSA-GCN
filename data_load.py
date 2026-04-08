import os
import random
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def setup_seed(seed: int):
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


setup_seed(1234)


def fea_load(species, dataset, k_value, q_th):
    nodes_name = np.load(
        f"datasets/{species}/PPI/{dataset}/gene_name.npy",
        allow_pickle=True
    )
    nodes_sum = len(nodes_name)

    # adjacency matrix
    A_static = np.load(f"datasets/{species}/PPI/{dataset}/SPIN.npy")
    A_sub = np.load(
        f"datasets/{species}/PPI/{dataset}/refined_net_and_fea/sub_dis_nul.npy"
    )
    A_orthologous = np.load(
        f"datasets/{species}/PPI/{dataset}/refined_net_and_fea/orth_dis_q={q_th}.npy"
    )

    # feature matrix
    fea_gene = np.load(
        f"datasets/{species}/PPI/{dataset}/refined_net_and_fea/fea_gene.npy"
    )
    fea_sub_1024_table = pd.read_csv(
        f"datasets/{species}/PPI/{dataset}/refined_net_and_fea/fea_sub_1024.csv"
    )
    fea_sub_1024 = fea_sub_1024_table.iloc[:, 2:].to_numpy()
    fea_orthologous = np.load(
        f"datasets/{species}/PPI/{dataset}/refined_net_and_fea/fea_orth.npy"
    )

    # labels
    label_y = torch.tensor(
        np.load(
            f"datasets/{species}/PPI/{dataset}/labels.npy"
        ).reshape(nodes_sum,),
        dtype=torch.float
    )

    return (
        nodes_sum,
        nodes_name,
        A_static,
        A_sub,
        A_orthologous,
        fea_gene,
        fea_sub_1024,
        fea_orthologous,
        label_y,
    )


def fea_load_coli(species, dataset, k_value, q_th):
    nodes_name = np.load(
        f"datasets/{species}/PPI/{dataset}/gene_name.npy",
        allow_pickle=True
    )
    nodes_sum = len(nodes_name)

    A_static = np.load(f"datasets/{species}/PPI/{dataset}/SPIN.npy")
    A_orthologous = np.load(
        f"datasets/{species}/PPI/{dataset}/refined_net_and_fea/orth_dis_q={q_th}.npy"
    )
    fea_gene = np.load(
        f"datasets/{species}/PPI/{dataset}/refined_net_and_fea/fea_gene.npy"
    )
    fea_orthologous = np.load(
        f"datasets/{species}/PPI/{dataset}/refined_net_and_fea/fea_orth.npy"
    )
    label_y = torch.tensor(
        np.load(
            f"datasets/{species}/PPI/{dataset}/labels.npy"
        ).reshape(nodes_sum,),
        dtype=torch.float
    )

    return (
        nodes_sum,
        nodes_name,
        A_static,
        A_orthologous,
        fea_gene,
        fea_orthologous,
        label_y,
    )


def infer_feature_dims(species, dataset):
    """
    Automatically infer real feature dimensions from saved preprocessing files.
    Returns:
        gene_dim, sub_dim, orth_dim
    """
    fea_gene = np.load(
        f"datasets/{species}/PPI/{dataset}/refined_net_and_fea/fea_gene.npy"
    )
    gene_dim = fea_gene.shape[1]

    fea_orth = np.load(
        f"datasets/{species}/PPI/{dataset}/refined_net_and_fea/fea_orth.npy"
    )
    orth_dim = fea_orth.shape[1]

    if species == "coli":
        sub_dim = 0
    else:
        fea_sub_table = pd.read_csv(
            f"datasets/{species}/PPI/{dataset}/refined_net_and_fea/fea_sub_1024.csv"
        )
        fea_sub = fea_sub_table.iloc[:, 2:].to_numpy()
        sub_dim = fea_sub.shape[1]

    return gene_dim, sub_dim, orth_dim


def load_gat_data(species, dataset, k_value, q_th):
    file_path = (
        f"datasets/{species}/PPI/{dataset}/data_and_model/"
        f"data_k={k_value:.2f}_q={q_th}.pt"
    )

    if os.path.exists(file_path):
        print(f"{file_path} The file exists.")
        data_gat_all = torch.load(file_path, weights_only=False)
    else:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        print(f"{file_path} The file does not exist.")
        (
            nodes_sum,
            nodes_name,
            A_static,
            A_sub,
            A_orthologous,
            fea_gene,
            fea_sub,
            fea_orthologous,
            label_y,
        ) = fea_load(species, dataset, k_value, q_th)

        print("fea_gene:", fea_gene.shape)
        print("fea_sub:", fea_sub.shape)
        print("fea_orthologous:", fea_orthologous.shape)

        feature_all = torch.tensor(
            np.hstack([fea_gene, fea_sub, fea_orthologous]),
            dtype=torch.float
        )

        # 2nd-layer graph
        source_index_2nd = []
        target_index_2nd = []
        for i in range(nodes_sum):
            source_index_2nd.append(i)
            target_index_2nd.append(i)
            for j in range(i + 1, nodes_sum):
                if A_sub[i, j] != 0:
                    source_index_2nd.append(i)
                    target_index_2nd.append(j)
                    source_index_2nd.append(j)
                    target_index_2nd.append(i)

        edge_index_2nd = torch.tensor(
            [source_index_2nd, target_index_2nd],
            dtype=torch.int64
        )
        dataset_2nd = Data(
            x=feature_all,
            y=label_y,
            edge_index=edge_index_2nd,
            num_classes=2
        )

        # 4th-layer graph
        source_index_4th = []
        target_index_4th = []
        for i in range(nodes_sum):
            source_index_4th.append(i)
            target_index_4th.append(i)
            for j in range(i + 1, nodes_sum):
                if A_orthologous[i, j] != 0:
                    source_index_4th.append(i)
                    target_index_4th.append(j)
                    source_index_4th.append(j)
                    target_index_4th.append(i)

        edge_index_4th = torch.tensor(
            [source_index_4th, target_index_4th],
            dtype=torch.int64
        )
        dataset_4th = Data(
            x=feature_all,
            y=label_y,
            edge_index=edge_index_4th,
            num_classes=2
        )

        data_gat_all = [dataset_2nd, dataset_4th]
        torch.save(data_gat_all, file_path)
        print("The file has been saved!")

    return data_gat_all


def load_gat_data_coli(species, dataset, k_value, q_th):
    file_path = (
        f"datasets/{species}/PPI/{dataset}/data_and_model/"
        f"data_k={k_value:.2f}_q={q_th}.pt"
    )

    if os.path.exists(file_path):
        print(f"{file_path} The file exists.")
        data_gat_all = torch.load(file_path, weights_only=False)
    else:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        print(f"{file_path} The file does not exist.")
        (
            nodes_sum,
            nodes_name,
            A_static,
            A_orthologous,
            fea_gene,
            fea_orthologous,
            label_y,
        ) = fea_load_coli(species, dataset, k_value, q_th)

        print("fea_gene:", fea_gene.shape)
        print("fea_orthologous:", fea_orthologous.shape)

        feature_all = torch.tensor(
            np.hstack([fea_gene, fea_orthologous]),
            dtype=torch.float
        )

        source_index_4th = []
        target_index_4th = []
        for i in range(nodes_sum):
            source_index_4th.append(i)
            target_index_4th.append(i)
            for j in range(i + 1, nodes_sum):
                if A_orthologous[i, j] != 0:
                    source_index_4th.append(i)
                    target_index_4th.append(j)
                    source_index_4th.append(j)
                    target_index_4th.append(i)

        edge_index_4th = torch.tensor(
            [source_index_4th, target_index_4th],
            dtype=torch.int64
        )
        dataset_4th = Data(
            x=feature_all,
            y=label_y,
            edge_index=edge_index_4th,
            num_classes=2
        )

        data_gat_all = [dataset_4th]
        torch.save(data_gat_all, file_path)
        print("The file has been saved!")

    return data_gat_all


def load_graph_data(args):
    """
    Unified graph data loading entry.
    """
    if args.species == "coli":
        return load_gat_data_coli(
            species=args.species,
            dataset=args.dataset,
            k_value=args.k_value,
            q_th=args.q_th,
        )
    return load_gat_data(
        species=args.species,
        dataset=args.dataset,
        k_value=args.k_value,
        q_th=args.q_th,
    )