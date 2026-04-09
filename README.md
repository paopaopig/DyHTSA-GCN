# DyHTSA-GCN

**Dynamic Hierarchical Temporal-Spatial-Attention Graph Convolutional Network for Essential Protein Prediction**
<p align="center"> <img src="https://github.com/user-attachments/assets/72870af4-7c0b-48f1-b9bb-bce4ba8fed4f" width="90%"> </p> <p align="center"> <img src="https://img.shields.io/badge/Python-3.8+-blue"> <img src="https://img.shields.io/badge/PyTorch-DeepLearning-red"> <img src="https://img.shields.io/badge/Status-Research-green"> </p>

📖 Overview

This repository provides the official implementation of DyHTSA-GCN, a novel multi-branch graph neural network designed for essential protein prediction.

The model integrates:

⏱️ Temporal dynamics (gene expression evolution)
🧭 Spatial information (subcellular localization)
🧬 Evolutionary signals (orthologous relationships)
🎯 Cross-layer attention mechanisms


📚 Table of Contents
Overview
Data Preprocessing
Training & Evaluation
Arguments
Usage
Reproducibility
Citation

🚀 1. Data Preprocessing

All preprocessing scripts are located in the `preprocessing/` directory.

🔧 Entry Point

#### `data_preprocess.py` (run first)

This script orchestrates the full preprocessing pipeline.

**Required arguments:**

- `species` — target organism  
- `dataset` — PPI dataset  
- `k_value` — threshold for dynamic network construction  
- `q_th` — threshold for orthologous information  

🧬 Step 1 — Gene & Label Construction

#### `name_spin_labels_process.py`

Generates:

- Protein (gene) list  
- Protein–protein interaction adjacency matrix (SPIN)  
- Essential protein labels  

🌐 Step 2 — Dynamic Network Construction

#### `dynamic_network_generate.py`

Constructs:

- Dynamic gene co-expression network  
- Gene expression feature matrix  

The dynamic network is controlled by parameter `k_value`.

🧭 Step 3 — Subcellular Features (Optional)

#### `sub_data_1024_process.py`

If subcellular localization data are available:

- Generates spatial feature matrix  

🧩 Step 4 — Multi-layer Network Construction

#### `multi-layer_network_construction.py`

Constructs:

- Spatial network  
- Evolutionary (orthologous) network  
- Corresponding feature matrices  

The orthologous network is filtered using `q_th`.

---

🧪 2. Training & Evaluation

🔧 Entry Point

#### `train_and_test.py`

Main script for:

- Model training  
- Cross-validation  
- Performance evaluation  

Pipeline includes:

1. Data loading  
2. Feature integration  
3. Model initialization  
4. Training  
5. Evaluation  

📦 Data Integration

#### `data_load.py`

Responsible for:

- Loading preprocessed features  
- Constructing multi-layer graph inputs  

🧠 Model Architecture

#### `model.py`

Implements the DyHTSA-GCN architecture:

- Dynamic branch (with TSA)  
- Spatial branch  
- Evolutionary branch  
- Cross-layer attention  
- Classification head  

---

⚙️ Command-line Arguments

All scripts support command-line arguments for flexible configuration.

❗ Core Parameters (Must Match)

The following parameters **must remain consistent between preprocessing and training**:

```text
--species
--dataset
--k_value
--q_th
```

Otherwise:

- files cannot be located  
- incorrect data may be loaded  

📌 Main Arguments


| Argument                |        Default | Description                                |
| ----------------------- | -------------: | ------------------------------------------ |
| `--species`             | `S.cerevisiae` | Species name                               |
| `--dataset`             |      `BIOGRID` | PPI dataset                                |
| `--k_value`             |         `3.00` | Threshold for dynamic network construction |
| `--q_th`                |             30 | Threshold for ortholog filtering           |
| `--k_folds`             |           `10` | Number of folds for cross-validation       |
| `--epochs_num`          |           `70` | Maximum training epochs                    |
| `--lr`                  |         `2e-3` | Learning rate                              |
| `--weight_decay`        |         `2e-4` | Weight decay                               |
| `--weight_positive`     |         `5.59` | Class imbalance weight                     |
| `--seed`                |           `45` | Random seed                                |
| `--drop_mlp`            |          `0.2` | MLP dropout                                |
| `--drop_conv`           |          `0.2` | GCN dropout                                |
| `--drop_classifier`     |          `0.2` | Classifier dropout                         |
| `--mlp_hidden`          |          `256` | MLP hidden dimension                       |
| `--classifier_hidden_1` |           `64` | Classifier layer 1                         |
| `--classifier_hidden_2` |           `32` | Classifier layer 2                         |
| `--conv_hidden1`        |           `64` | First GCN layer                            |
| `--conv_hidden2`        |           `32` | Second GCN layer                           |
| `--T`                   |            `5` | Temporal window size                       |
| `--lambda_adv`          |          `0.1` | Adversarial loss weight                    |
| `--epsilon`             |          `0.1` | Adversarial perturbation                   |
| `--early_stop_patience` |            `5` | Early stopping                             |

| Argument | Default | Description |
|---|---:|---|
| `--species` | `S.cerevisiae` | Species name |
| `--dataset` | `BIOGRID` | PPI dataset |
| `--k_value` | `3.00` | Threshold for dynamic network construction |
| `--q_th` | 30 | Threshold for ortholog filtering |
| `--k_folds` | `10` | Number of folds for cross-validation |
| `--epochs_num` | `70` | Maximum training epochs |
| `--lr` | `2e-3` | Learning rate |
| `--weight_decay` | `2e-4` | Weight decay |
| `--weight_positive` | `5.59` | Class imbalance weight |
| `--seed` | `45` | Random seed |
| `--drop_mlp` | `0.2` | MLP dropout |
| `--drop_conv` | `0.2` | GCN dropout |
| `--drop_classifier` | `0.2` | Classifier dropout |
| `--mlp_hidden` | `256` | MLP hidden dimension |
| `--classifier_hidden_1` | `64` | Classifier layer 1 |
| `--classifier_hidden_2` | `32` | Classifier layer 2 |
| `--conv_hidden1` | `64` | First GCN layer |
| `--conv_hidden2` | `32` | Second GCN layer |
| `--T` | `5` | Temporal window size |
| `--lambda_adv` | `0.1` | Adversarial loss weight |
| `--epsilon` | `0.1` | Adversarial perturbation |
| `--early_stop_patience` | `5` | Early stopping |


---

⚡ Quick Start

🧬 Default (S.cerevisiae — Paper Setting)

```bash
python preprocessing/data_preprocess.py \
    --species S.cerevisiae \
    --dataset BIOGRID \
    --k_value 3.00 \
    --q_th 30

python train_and_test.py \
    --species S.cerevisiae \
    --dataset BIOGRID \
    --k_value 3.00 \
    --q_th 30
```

🪰 Example (Drosophila melanogaster)

```bash
python preprocessing/data_preprocess.py \
    --species D.melanogaster \
    --dataset DIP \
    --k_value 3.00 \
    --q_th 30

python train_and_test.py \
    --species D.melanogaster \
    --dataset DIP \
    --k_value 3.00 \
    --q_th 30
```

---

🔁 Reproducibility Notes

- Preprocessing must be completed before training  
- All parameters must remain consistent  
- Random seed should be fixed  
- Generated files depend on parameters  

Example generated files:

```text
dynamic_network_k=3.00_all_fea_(1-F).pt
data_k=3.00_q=30.pt
```

---

⭐ Summary

The full workflow is:

```text
Preprocessing → Multi-layer Network Construction → Training → Evaluation
```
