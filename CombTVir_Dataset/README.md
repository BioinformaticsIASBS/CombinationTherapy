# **The CombTVir_Dataset details**
* **“Comb_DDV.csv” contains the info of drug_drug_virus interactions.**
* **“drug_dict.txt” contains the drugs’ names and their corresponding DrugBank IDs.**
* **“virus_dict.txt” contains the viruses’ names and their corresponding NCBI IDs.**
* **“Virus_fasta.fasta” contains the viruses’ sequences in FASTA format.**
* **“Drugs_info.csv” is a table in which each row drug name, its corresponding DrugBank ID, its SMILES, and generated Tanimoto similarities with all drugs in the dataset.**
* **“Xindex.npy” is a Python binary data file that contains all Cartesian products of drug×drug× virus (including positive and negative combinations.)**
* **“Y.npy” contains the corresponding label of each combination in Xindex.npy.** 
* **“Similarity_Matrix_Drugs.txt” contains the drug’s Tanimato similarities and is arranged in drug_dict file order.** 
* **“Similarity_Matrix_Viruses” contains virus sequence alignment score and is arranged in the order of the virus_dict file.**
