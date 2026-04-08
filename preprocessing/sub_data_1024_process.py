import pandas as pd
import numpy as np
import os

def process_subcellular_features(species, dataset):

    txt_path = f"../datasets/{species}/SubLocalizations/{species}_compartment_integrated_full.txt"
    output_prefix = f"../datasets/{species}/PPI/{dataset}/refined_net_and_fea/fea_sub_1024.csv"
    gene_names_path = f"../datasets/{species}/PPI/{dataset}/gene_name.npy"
    uniprot_map_path = f"../datasets/{species}/PPI/{dataset}/uniprotid_to_gene.xlsx"

    if not os.path.exists(output_prefix):
        os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
        print(f"loading: {txt_path}")
        df = pd.read_csv(txt_path, sep='\t', header=None, names=["esm_id", "protein_id", "go_id", "location", "score"])

        print("Count the number of proteins that appear in each subcellular location...")
        location_counts = (
            df.groupby("location")["protein_id"]
            .nunique()
            .sort_values(ascending=False)
        )

        top_locations = location_counts.head(1024).index.tolist()
        df = df[df['location'].isin(top_locations)]

        print("Construct the protein × position score matrix...")
        pivot_table = df.pivot_table(
            index="protein_id",
            columns="location",
            values="score",
            aggfunc="max",
            fill_value=0
        )

        pivot_table = pivot_table[top_locations]
        print("The first step is completed! The txt file has been extracted successfully! Dimension =", pivot_table.shape)

        uniprot_ids = np.load(gene_names_path, allow_pickle=True)
        uniprot_ids = [str(uid).strip() for uid in uniprot_ids]

        mapping_df = pd.read_excel(uniprot_map_path)
        mapping_df["From"] = mapping_df["From"].astype(str).str.strip()
        mapping_df["To"] = mapping_df["To"].astype(str).str.strip()

        from_to_map = {}
        for _, row in mapping_df.iterrows():
            uid = row["From"].strip()
            if uid and uid not in from_to_map:
                from_to_map[uid] = row["To"]

        mapped_gene_names = []
        for raw_entry in uniprot_ids:
            mapped_gene = None
            for sub_id in str(raw_entry).split('|'):
                sub_id = sub_id.strip()
                if sub_id in from_to_map:
                    mapped_gene = from_to_map[sub_id]
                    break
            mapped_gene_names.append(mapped_gene)

        pivot_table.index = pivot_table.index.astype(str).str.strip()
        pivot_index_set = set(pivot_table.index)

        aligned_rows = []
        for uid, gene in zip(uniprot_ids, mapped_gene_names):
            if gene is not None and gene in pivot_index_set:
                aligned_rows.append(pivot_table.loc[gene].values)
            else:
                aligned_rows.append([0] * pivot_table.shape[1])

        aligned_df = pd.DataFrame(aligned_rows, columns=pivot_table.columns)
        aligned_df.insert(0, "UniProt_ID", uniprot_ids)
        aligned_df.insert(1, "Gene_Name", mapped_gene_names)

        print(f"Step Two is done! Final dimension = {aligned_df.shape}")
        aligned_df.to_csv(output_prefix, index=False)
        print(f"Finished fea_sub_1024.csv!")

        num_mapped = sum(gene is not None for gene in mapped_gene_names)
        num_gene_in_features = sum((gene is not None and gene in pivot_index_set) for gene in mapped_gene_names)
        print(f"The number successfully mapped to gene name: {num_mapped} / {len(uniprot_ids)}")
        print(f"The number of genes successfully matched to the feature table: {num_gene_in_features} / {num_mapped}")
    else:
        print(f"✅ fea_sub_1024.csv already exists, skipped saving!")

