import numpy as np
import pandas as pd
import os

def construct_of_2nd_layer(nodes_names, SPIN, species, dataset, data_file):
    sublocation = pd.read_csv(data_file)
    gene_name = np.array(sublocation.iloc[:, 0])
    subcell_nuc = sublocation["Nucleus"].to_numpy().reshape(-1, 1)
    subcell_nuc = (subcell_nuc != 0).astype(float)

    nodes_nucleus = []
    for i in range(len(nodes_names)):
        if nodes_names[i] in gene_name:
            ind = np.argwhere(gene_name == nodes_names[i])
            nodes_nucleus.append(subcell_nuc[ind[0, 0], :])
        else:
            nodes_nucleus.append(np.zeros(subcell_nuc.shape[1]))
    nodes_nucleus = np.array(nodes_nucleus)
    SUB_PIN = np.zeros((len(nodes_names), len(nodes_names)))
    for i in range(len(nodes_names)):
        for j in range(i + 1, len(nodes_names)):
            if (nodes_nucleus[i,:] * nodes_nucleus[j,:]).sum() >= 1:
                SUB_PIN[i, j] = SUB_PIN[j, i] = 1
    SUB_PIN = SUB_PIN * SPIN
    print("The number of interactions in the spatial network: ", sum(sum(SUB_PIN)) / 2)
    return SUB_PIN

def construct_of_4th_layer(nodes_names, SPIN, species, dataset, data_file, threshold):
    orthologs_data = pd.read_csv(data_file)
    gene_name = np.array(orthologs_data['Gene'])
    gene_orthologs_values = np.array(orthologs_data.iloc[:, :-1])

    nodes_orthologs_value = []
    for node in nodes_names:
        if node in gene_name:
            ind = np.where(gene_name == node)[0]
            nodes_orthologs_value.append(gene_orthologs_values[ind[0], :])
        else:
            nodes_orthologs_value.append(np.zeros(gene_orthologs_values.shape[1]))

    nodes_orthologs_value = np.array(nodes_orthologs_value)
    nodes_orthologs_score = np.sum(nodes_orthologs_value, axis=1)
    fea_orth_save_path = f"../datasets/%s/PPI/%s/refined_net_and_fea/fea_orth.npy"%(species, dataset)
    if not os.path.exists(fea_orth_save_path):
        os.makedirs(os.path.dirname(fea_orth_save_path), exist_ok=True)
        np.save(fea_orth_save_path, nodes_orthologs_value)
        print(f"Finished fea_orth.npy!")
    else:
        print(f"fea_orth.npy already exists, skipped saving!")
    ORTH_PIN = np.zeros((len(nodes_names), len(nodes_names)))
    for i in range(len(nodes_names)):
        for j in range(i + 1, len(nodes_names)):
            if ((nodes_orthologs_value[i, :] * nodes_orthologs_value[j, :]).sum() >= 1 and
                nodes_orthologs_score[i] >= threshold and nodes_orthologs_score[j] >= threshold):
                ORTH_PIN[i, j] = ORTH_PIN[j, i] = 1
    ORTH_PIN = ORTH_PIN * SPIN
    print("The number of interactions in the evolutionary network: ",sum(sum(ORTH_PIN)) / 2)
    return ORTH_PIN

def obtain_2nd_and_4th_fea_and_dis(species, dataset, q_th):
    nodes_names = np.load(r"../datasets/%s/PPI/%s/gene_name.npy" % (species, dataset), allow_pickle=True)
    SPIN = np.load(r'../datasets/%s/PPI/%s/SPIN.npy' % (species, dataset), allow_pickle=True)

    sub_data_path = "../datasets/%s/PPI/%s/refined_net_and_fea/fea_sub_1024.csv"%(species, dataset)
    sub_dis_save_path = f"../datasets/%s/PPI/%s/refined_net_and_fea/sub_dis_nul.npy"%(species,dataset)
    if species!="coli":
        if not os.path.exists(sub_dis_save_path):
            os.makedirs(os.path.dirname(sub_dis_save_path), exist_ok=True)
            the_2nd_layer = construct_of_2nd_layer(nodes_names, SPIN, species, dataset, sub_data_path)
            np.save(sub_dis_save_path, the_2nd_layer)
            print(f"Finished sub_dis.npy!")
        else:
            print(f"sub_dis.npy already exists, skipped saving!")

    orth_data_path = "../datasets/%s/Orthologs/orthologs.csv"%(species)
    orth_dis_save_path = f"../datasets/%s/PPI/%s/refined_net_and_fea/orth_dis_q=%d.npy"%(species, dataset, q_th)
    if not os.path.exists(orth_dis_save_path):
        os.makedirs(os.path.dirname(orth_dis_save_path), exist_ok=True)
        the_4th_layer = construct_of_4th_layer(nodes_names, SPIN, species, dataset, orth_data_path, threshold=q_th)
        np.save(orth_dis_save_path, the_4th_layer)

        print(f"✅✅✅ Finished orth_dis.npy!")
    else:
        print(f"✅ orth_dis.npy already exists, skipped saving!")


