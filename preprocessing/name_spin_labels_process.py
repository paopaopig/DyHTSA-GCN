import pandas as pd
import numpy as np
import os

def name_spin_labels_process(species,dataset):
    species = species
    dataset = dataset
    print("The current species is {species}, and the dataset is {dataset}.".format(species=species, dataset=dataset))
    # 1. gene_name
    gene_name_save_path = f"../datasets/%s/PPI/%s/gene_name.npy"%(species,dataset)
    if not os.path.exists(gene_name_save_path):
        file_path = r"../datasets/%s/PPI/%s/%s.csv"%(species,dataset,dataset.lower())
        df = pd.read_csv(file_path, header=0, names=["protein1", "protein2"])

        # Delete rows containing null values or "blank"
        def is_valid(protein):
            return pd.notna(protein) and str(protein).strip() != "" and str(protein).strip().lower() != "blank"
        original_len = len(df)
        df = df[df["protein1"].apply(is_valid) & df["protein2"].apply(is_valid)]
        cleaned_len = len(df)
        print(f"Number of steps before cleaning: {original_len}, after cleaning {cleaned_len}")
        print(df.head())

        proteins = pd.unique(df[['protein1', 'protein2']].values.ravel())
        proteins = np.sort(proteins)
        print("The number of proteins",len(proteins))
        os.makedirs(os.path.dirname(gene_name_save_path), exist_ok=True)
        np.save(gene_name_save_path, proteins)
        print(f"Finished gene_name.npy!")
    else:
        print(f"gene_name.npy already exists, skipped saving!")
        gene_name = np.load(gene_name_save_path, allow_pickle=True)
        print(f"The number of proteins: {gene_name.shape}")
    gene_name_save_path_csv = f"../datasets/%s/PPI/%s/gene_name.csv" % (species, dataset)
    if not os.path.exists(gene_name_save_path_csv):
        os.makedirs(os.path.dirname(gene_name_save_path_csv), exist_ok=True)
        proteins = np.load(gene_name_save_path, allow_pickle=True)
        pd.DataFrame(proteins, columns=["Protein"]).to_csv(gene_name_save_path_csv, index=False)
        print(f"Finished gene_name.csv!")
    else:
        print(f"gene_name.csv already exists, skipped saving!")

    # 2. SPIN
    SPIN_save_path = f"../datasets/%s/PPI/%s/SPIN.npy"%(species,dataset)
    if not os.path.exists(SPIN_save_path):
        protein_to_index = {protein: idx for idx, protein in enumerate(proteins)}
        N = len(proteins)
        adj_matrix = np.zeros((N, N))
        for _, row in df.iterrows():
            p1, p2 = row['protein1'], row['protein2']
            i, j = protein_to_index[p1], protein_to_index[p2]
            if i != j:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
        print(adj_matrix)
        print("The number of interactions：",sum(sum(adj_matrix))/2)
        os.makedirs(os.path.dirname(SPIN_save_path), exist_ok=True)
        np.save(SPIN_save_path, adj_matrix)
        print(f"Finished SPIN.npy!")
    else:
        print(f"SPIN.npy already exists, skipped saving!")

    # 2. labels
    labels_save_path = f"../datasets/%s/PPI/%s/labels.npy"%(species,dataset)
    if not os.path.exists(labels_save_path):
        label_df = pd.read_csv("../datasets/%s/EssentialGenes/ogee.csv"%(species))
        gene_to_label = dict(zip(label_df['Gene'], label_df['Label']))

        labels = np.array([gene_to_label.get(protein, 0) for protein in proteins], dtype=np.uint8)
        print("The number of essential proteins is:",labels.sum())
        os.makedirs(os.path.dirname(labels_save_path), exist_ok=True)
        np.save(labels_save_path, labels)
        print(f"Finished labels.npy!")
    else:
        labels = np.load(labels_save_path)
        print(f"labels.npy already exists, skipped saving!")
        print(f"The number of essential proteins: {sum(labels)}")
