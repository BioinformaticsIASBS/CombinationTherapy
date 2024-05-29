# **Data**
All necessary data for training ML and DRaW models are stored in the "Data" folder, which is further divided into ML and DRaW folders. Each essential file will be introduced in the following lines:
* **Similarity_Matrix_Drugs:** It contains similarity scores between each paired drug that are calculated using "Tanimoto" similarity.
* **Similarity_Matrix_Viruses:** It contains similarity scores between each paired virus that are calculated using sequence alignment.
* **Xindex.npy:** "It includes all potential combinations, comprising of two antivirals and a virus."
* **Y.npy:** This file includes the labels that match each of the compounds in the "Xindex.npy" file.
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
```bash
@article{majidifar2024combination,
  title={Combination Therapy Synergism Prediction for Virus Treatment Using Machine and Deep Learning Models},
  author={Majidifar, Shayan and Zabihian, Arash and Hooshmand, Mohsen},
  year={2024},
  url={https://doi.org/10.21203/rs.3.rs-4389305/v1}
}
```

