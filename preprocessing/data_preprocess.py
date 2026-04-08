from name_spin_labels_process import name_spin_labels_process
from dynamic_network_generate import dy_net_construction
from multi_layer_network_construction import obtain_2nd_and_4th_fea_and_dis
from sub_data_1024_process import process_subcellular_features

species = "S.cerevisiae"
dataset = "BIOGRID"
k_value = 3.00
q_th = 30

##### Data preprocessing ######
# Step 1: Get gene_name,SPIN,labels
name_spin_labels_process(species = species, dataset = dataset)

# Step 2: Obtain Gene_expression_feature and dynamic_network
dy_net_construction(species = species, dataset = dataset, k_value = k_value)

# Step 3: Obtain Subcellular_feature,Orthologous_feature,Subcellular_network,and Orthologous_network
if species!='coli':
    process_subcellular_features(species = species, dataset = dataset)
obtain_2nd_and_4th_fea_and_dis(species = species, dataset = dataset, q_th = q_th)
