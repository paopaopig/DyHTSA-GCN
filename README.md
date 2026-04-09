# DyHTSA-GCN

**Dynamic Hierarchical Temporal-Spatial-Attention Graph Convolutional Network for Essential Protein Prediction**

## Pipeline Overview
<img width="4095" height="2345" alt="3aa7f0f768cc56ca7680638f1ffef7b4" src="https://github.com/user-attachments/assets/72870af4-7c0b-48f1-b9bb-bce4ba8fed4f" />


This project provides the complete pipeline for reproducing **DyHTSA-GCN**, including:

- Data preprocessing  
- Multi-layer network construction  
- Model training and evaluation  

The workflow is divided into two main stages:

---

## 1. Data Preprocessing

All preprocessing scripts are located in the `preprocessing/` directory.

### Entry Point

#### `data_preprocess.py` (run first)

This script orchestrates the full preprocessing pipeline.

**Required arguments:**

- `species` — target organism  
- `dataset` — PPI dataset  
- `k_value` — threshold for dynamic network construction  
- `q_th` — threshold for orthologous information  

### Step 1 — Gene and Label Construction

#### `name_spin_labels_process.py`

Generates:

- Protein (gene) list  
- Protein–protein interaction adjacency matrix (SPIN)  
- Essential protein labels  

### Step 2 — Dynamic Network Construction

#### `dynamic_network_generate.py`

Constructs:

- Dynamic gene co-expression network  
- Gene expression feature matrix  

The dynamic network is controlled by parameter `k_value`.

### Step 3 — Subcellular Feature Construction (Optional)

#### `sub_data_1024_process.py`

If subcellular localization data are available:

- Generates spatial feature matrix  

### Step 4 — Multi-layer Network Construction

#### `multi-layer_network_construction.py`

Constructs:

- Spatial network  
- Evolutionary (orthologous) network  
- Corresponding feature matrices  

The orthologous network is filtered using `q_th`.

---

## 2. Training, Evaluation, and Model

### Entry Point

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

### Data Integration

#### `data_load.py`

Responsible for:

- Loading preprocessed features  
- Constructing multi-layer graph inputs  

### Model Definition

#### `model.py`

Implements the DyHTSA-GCN architecture:

- Dynamic branch (with TSA)  
- Spatial branch  
- Evolutionary branch  
- Cross-layer attention  
- Classification head  

---

## Command-line Arguments

All scripts support command-line arguments for flexible configuration.

### Core Parameters (Must Match Preprocessing)

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

### Main Arguments

<<<<<<< HEAD
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
=======
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
>>>>>>> ba0bc46 (first commit)

---

## Example Usage

### Default (Yeast — Paper Setting)

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

### Example (Drosophila melanogaster)

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

## Reproducibility Notes

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

## Summary

The full workflow is:

```text
Preprocessing → Multi-layer Network Construction → Training → Evaluation
```
