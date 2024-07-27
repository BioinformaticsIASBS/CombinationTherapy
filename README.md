# Combination Therapy Synergism Prediction for Virus Treatment Using Machine Learning Models
## by Shayan Majidifar, Arash Zabihian, Mohsen Hooshmand

# **Data**
All necessary data for training ML and DRaW models are stored in the "CombTVir_Dataset" folder, which is further divided into ML and DRaW folders. The dataset contains:
* **“Comb_DDV.csv” contains the info of drug_drug_virus interactions.**
* **“drug_dict.txt” contains the drugs’ names and their corresponding DrugBank IDs.**
* **“virus_dict.txt” contains the viruses’ names and their corresponding NCBI IDs.**
* **“Virus_fasta.fasta” contains the viruses’ sequences in FASTA format.**
* **“Drugs_info.csv” is a table in which each row drug name, its corresponding DrugBank ID, its SMILES, and generated Tanimoto similarities with all drugs in the dataset.**
* **“Xindex.npy” is a Python binary data file that contains all Cartesian products of drug×drug× virus (including positive and negative combinations.)**
* **“Y.npy” contains the corresponding label of each combination in Xindex.npy.** 
* **“Similarity_Matrix_Drugs.txt” contains the drug’s Tanimato similarities and is arranged in drug_dict file order.** 
* **“Similarity_Matrix_Viruses” contains virus sequence alignment score and is arranged in the order of the virus_dict file.**
# **Train DRaW and ML models**
In our experiments, we utilized Python 3.9.4.
## **ML methods**
To train ML models, you need to follow these steps:
1. Create ratioed data using the command for all ratios that include 1:3, 1:5, 1:10, 1:100, and 1:500. <br>
```bash
python create_ratioed_data_ML.py
```
2. Use the following command to apply SVM and RF on the ratioed data with ML.py script. <be>
```bash
python ML.py
```
## **DRaW method**
To train the DRaW model, you can use the following command as an example: <be>
```bash
python DRaW.py --data_path DRaW --ratio 3 --result_path DRaW_results/model
```

## **Citation**
```
@article{majidifar2024combination,
  title={Combination Therapy Synergism Prediction for Virus Treatment Using Machine Learning Models},
  author={Majidifar, Shayan and Zabihian, Arash and Hooshmand, Mohsen},
  year={2024},
  url={https://doi.org/10.21203/rs.3.rs-4389305/v1}
}
```

