# Project Overview

This repository contains code and notebooks to preprocess structured and unstructured datasets, and to predict short-term mortality and readmission within 30 days using different models. The processing is divided into three main parts:

1. **Preprocessing**  
2. **Prediction on Structured Data**  
3. **Prediction on Unstructured Data**  
4. **Multimodal Approach**

---

## 1. Preprocessing

- **Structured Data:**  
  - Notebook: `FinalCode/Structure_preprocessig_final.ipynb`  
  - Purpose: Create and preprocess the structured dataset.

- **Unstructured Data:**  
  - Notebook: `FinalCode/Unstructure_preproceesing_final.ipynb`  
  - Purpose: Create and preprocess the unstructured dataset.

---

## 2. Prediction on Structured Data

- **Model:** BEHRT  
- **File:** `FinalCode/BEHRT.ipynb`  
- **Details:**  
  - This notebook uses the BEHRT model to predict short-term mortality and readmission within 30 days on the structured dataset.  
  - The results for this prediction are already included in the notebook.

---

## 3. Prediction on Unstructured Data

- **Model:** BioClinicalBERT  
- **File:** `FinalCode/04_BioclinicalBERT.py`  
- **Details:**  
  - This script processes the unstructured dataset using the BioClinicalBERT model to predict the same outcomes (short-term mortality and readmission).  
  - The output of this process is saved in the file `04_bio.log`.

---

## 4. Multimodal Approach

- **File:** `FinalCode/05_Multimodal_Average_Fusion.py`  
- **Details:**  
  - This script combines both structured and unstructured datasets to create a multimodal prediction model.  
  - The results from this run are stored in `05_average.log`.

---

## How to Use

1. **Run Preprocessing:**  
   Execute the preprocessing notebooks to generate the datasets.
   - For structured data: Run `Structure_preprocessing_final.ipynb`
   - For unstructured data: Run `Unstructure_preproceesing_final.ipynb`

2. **Run Predictions:**
   - For structured data, review the results in `BEHRT.ipynb`.
   - For unstructured data, run the script `BioclinicalBert.py` and check the output in `unstruct.log`.

3. **Multimodal Prediction:**
   - Execute `Multimodal.py` to run the multimodal prediction.  
   - Check the output in `multi.log`.


