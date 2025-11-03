# Muscle Features Analysis
Python framework for muscle feature analysis and visualization using statistical testing and demographic stratification. (Data are internal and excluded for privacy and institutional policy reasons.)

⚠️ **Note:**  
The feature extraction and dataset used in this analysis are internal and not shared here 
due to institutional and privacy restrictions.  
This repository focuses solely on the analysis workflow and visualization pipeline.

## 🧩 Overview

The main analysis script performs the following:
- Merges feature and patient information tables  
- Groups subjects by age and gender  
- Computes group statistics using the Mann–Whitney U test with Bonferroni correction  
- Generates boxplots for muscle feature distributions (volume, lean mass, mean HU, fat ratio)  

All steps are fully modular — you can adapt the paths and column names to your own dataset.


